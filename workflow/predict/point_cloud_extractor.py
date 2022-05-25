"""
Copyright (c) 2020 CSi Biotech
3D-MSNet is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
"""

import os
import csv
import time
import glob
import math
import argparse

import torch
import numpy as np
import msnet_cuda as msnet

from pyteomics import mzml, mzxml
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED


class Spectrum:
    def __init__(self, rt, mzs, intensities):
        self.rt = rt
        self.mzs = mzs
        self.intensities = intensities


class _BasePCGenerator:
    def __init__(self, ms_file_path, output_dir):
        if ms_file_path.lower().endswith('.mzml'):
            self.ms_file = mzml.read(ms_file_path)
        else:
            # TODO mzxml test
            assert ms_file_path.lower().endswith('.mzxml')
            self.ms_file = mzxml.read(ms_file_path)
        self.output_dir = output_dir
        self.ms1_spectra = []
        self.ms2_spectra = {}

    def load_ms1_spectra(self, min_intensity):
        max_length = 0
        for spectrum in self.ms_file:
            if spectrum.get('ms level') != 1:
                continue
            mzs = spectrum.get('m/z array')
            intensities = spectrum.get('intensity array')
            rt = spectrum.get('scanList').get('scan')[0].get('scan start time')
            idx = intensities > min_intensity
            mzs = mzs[idx]
            intensities = intensities[idx]
            self.ms1_spectra += [Spectrum(rt, mzs, intensities)]
            if len(mzs) > max_length:
                max_length = len(mzs)
        print(max_length)

    def load_ms2_spectra(self, min_intensity):
        for spectrum in self.ms_file:
            if spectrum.get('ms level') != 2:
                continue
            iso_window = spectrum.get('precursorList').get('precursor')[0].get('isolationWindow')
            precursor_mz = iso_window.get('isolation window target m/z')
            mz_win_lower = precursor_mz - iso_window.get('isolation window lower offset')
            mz_win_upper = precursor_mz + iso_window.get('isolation window upper offset')

            mzs = spectrum.get('m/z array')
            intensities = spectrum.get('intensity array')
            idx = intensities > min_intensity
            mzs = mzs[idx]
            intensities = intensities[idx]
            rt = spectrum.get('scanList').get('scan')[0].get('scan start time')
            if self.ms2_spectra.get((mz_win_lower, mz_win_upper)) is None:
                self.ms2_spectra[(mz_win_lower, mz_win_upper)] = [Spectrum(rt, mzs, intensities)]
            else:
                self.ms2_spectra[(mz_win_lower, mz_win_upper)] += [Spectrum(rt, mzs, intensities)]

    def write_to_csv(self, pcs, target_mzs, target_rts=None, ids=None):
        def write(output_dir, ordinal, mz, rt, data):
            file_name = '{}_{}_{}.csv'.format(ordinal, mz, rt)
            output_file = open(os.path.join(output_dir, file_name), 'w')
            writer = csv.writer(output_file)
            writer.writerows(data)
            output_file.close()
            if ordinal % 1000 == 0:
                print(file_name)

        executor = ThreadPoolExecutor()
        tasks = []
        for i, pc in enumerate(pcs):
            if len(pc) < 10:
                continue
            if target_rts is None:
                target_rt = '-'
            else:
                target_rt = format(target_rts[i], '.2f')
            if ids is None:
                id_num = i + 1
            else:
                id_num = ids[i]
            task = executor.submit(write, self.output_dir, id_num, format(target_mzs[i], '.4f'), target_rt, pc)
            tasks += [task]
        wait(tasks, return_when=ALL_COMPLETED)
        print("Write finished.")


class _Extractor:
    def _extract_pc(self, spectra, from_rt, to_rt, from_mz, to_mz):
        pc = []
        for spectrum in spectra:
            if spectrum.rt <= from_rt:
                continue
            if spectrum.rt >= to_rt:
                break

            filtered_idx = (spectrum.mzs >= from_mz) * (spectrum.mzs <= to_mz)
            mzs = spectrum.mzs[filtered_idx].astype(np.float32)
            intensities = spectrum.intensities[filtered_idx].astype(np.float32)
            rts = np.ones(len(mzs), dtype=np.float32) * spectrum.rt

            pc += [np.vstack((rts, mzs, intensities)).transpose()]
        if len(pc) > 0:
            return np.concatenate(pc)
        else:
            return np.array([])

    def _bat_extract_pc(self, spectra, target_mzs, target_rts=None, rt_tolerance=3.0, mz_tolerance=0.4):
        bat_pc = []
        for i, mz in enumerate(target_mzs):
            from_mz = mz - mz_tolerance
            to_mz = mz + mz_tolerance
            if target_rts is not None:
                from_rt = target_rts[i] - rt_tolerance
                to_rt = target_rts[i] + rt_tolerance
            else:
                from_rt = 0
                to_rt = 100000
            pc = self._extract_pc(spectra, from_rt, to_rt, from_mz, to_mz)
            bat_pc += [pc]
        return bat_pc

    def _extract_pc_fast(self, min_intensity, rt_tolerance, mz_tolerance, target_rt, target_mz, full_points):
        n = len(full_points)
        rt_len = len(target_rt)
        mz_len = len(target_mz)
        idx = torch.cuda.IntTensor(4, n).fill_(-1)
        msnet.extract_pc_wrapper(n, rt_len, mz_len, min_intensity, rt_tolerance, mz_tolerance,
                                 torch.tensor(target_rt, dtype=torch.float32).cuda(),
                                 torch.tensor(target_mz, dtype=torch.float32).cuda(),
                                 torch.tensor(full_points, dtype=torch.float32).cuda().contiguous(), idx.cuda())
        return idx

    def _bat_extract_pc_fast(self, spectra, target_mz, target_rt=None, rt_tolerance=3.0, mz_tolerance=0.4):
        # prepare points
        start_time = time.time()
        full_points = []
        for i in range(len(spectra)):
            spectrum = spectra[i]

            mzs = spectrum.mzs.astype(np.float32)
            intensities = spectrum.intensities.astype(np.float32)
            rts = np.ones(len(mzs), dtype=np.float32) * spectrum.rt

            full_points += [np.vstack((rts, mzs, intensities)).transpose()]
        full_points = np.concatenate(full_points)
        print("--prepare points: " + str(time.time() - start_time))
        start_time = time.time()

        # blocks
        if target_rt is None:
            target_rt = np.zeros(target_mz.shape)
            rt_tolerance = 100000

        block_idx = self._extract_pc_fast(0, rt_tolerance, mz_tolerance, target_rt, target_mz, full_points).cpu().numpy()
        print("--extract: " + str(time.time() - start_time))
        start_time = time.time()

        bat_pc = []
        for i in range(len(target_rt) * len(target_mz)):
            bat_pc += [[]]
        for i in range(4):
            for j in range(len(full_points)):
                tmp_idx = block_idx[i, j]
                if tmp_idx == -1:
                    continue
                bat_pc[tmp_idx] += [full_points[j]]

        print("--assembly: " + str(time.time() - start_time))
        return bat_pc


class DDATargetGenerator(_BasePCGenerator, _Extractor):
    def __init__(self, ms_file_path, output_dir, lib_path, rt_tolerance, mz_tolerance, min_intensity=128):

        super().__init__(ms_file_path=ms_file_path, output_dir=output_dir)
        self.rt_tolerance = rt_tolerance
        self.mz_tolerance = mz_tolerance

        self.load_csv_lib(lib_path)
        self.load_ms1_spectra(min_intensity)

    def load_csv_lib(self, lib_path):
        lib_file = open(lib_path, 'r')
        reader = csv.reader(lib_file)
        header = next(reader)
        self.lib = {col_name: [] for col_name in header}
        for line in reader:
            for i in range(len(line)):
                self.lib[header[i]] += [line[i]]

    def generate(self):
        assert self.lib.__contains__('m/z')
        target_mzs = np.array(self.lib['m/z']).astype(np.float32)
        if self.lib.__contains__('RT(min)'):
            target_rts = np.array(self.lib['RT(min)']).astype(np.float32)
        else:
            target_rts = None

        start_time = time.time()
        bat_pc = self._bat_extract_pc(self.ms1_spectra, target_mzs, target_rts,
                                      self.rt_tolerance, self.mz_tolerance)
        print('extract time:' + str(time.time() - start_time))
        # bat_pc = self._bat_extract_pc_fast(self.ms1_spectra, target_mzs, target_rts,
        #                                    self.rt_tolerance, self.mz_tolerance)
        self.write_to_csv(bat_pc, target_mzs, target_rts)
        print(self.output_dir + ' generated')


class DDAUntargetGenerator(_BasePCGenerator, _Extractor):
    def __init__(self, ms_file_path, output_dir, from_mz, to_mz, from_rt, to_rt, window_mz_width=0.8, window_rt_width=6,
                 max_peak_mz_width=0.1, max_peak_rt_width=1, min_intensity=128):
        super().__init__(ms_file_path=ms_file_path, output_dir=output_dir)
        self.from_mz = from_mz
        self.to_mz = to_mz
        self.from_rt = from_rt
        self.to_rt = to_rt
        self.window_mz_width = window_mz_width
        self.window_rt_width = window_rt_width
        self.max_peak_mz_width = max_peak_mz_width
        self.max_peak_rt_width = max_peak_rt_width

        self.load_ms1_spectra(min_intensity)

    def generate(self):
        mz_frags = math.ceil(float(self.to_mz - self.from_mz) / self.window_mz_width)
        rt_frags = math.ceil(float(self.to_rt - self.from_rt) / self.window_rt_width)
        print('Fragmenting to {} point clouds'.format(mz_frags * rt_frags))

        center_mzs = np.arange(0.5, mz_frags) * self.window_mz_width + self.from_mz
        center_rts = np.arange(0.5, rt_frags) * self.window_rt_width + self.from_rt

        window_rts, window_mzs = np.meshgrid(center_rts, center_mzs)
        window_rts = window_rts.flatten().astype(np.float32)
        window_mzs = window_mzs.flatten().astype(np.float32)

        mz_tolerance = self.window_mz_width / 2.0 + self.max_peak_mz_width
        rt_tolerance = self.window_rt_width / 2.0 + self.max_peak_rt_width

        start_time = time.time()
        # target_mzs = np.array([684.0])
        # center_mzs = np.array([500.0])
        # center_rts = np.array([90.0])
        # mz_tolerance = 100
        # rt_tolerance = 20
        bat_pc = self._bat_extract_pc_fast(self.ms1_spectra, center_mzs, center_rts, rt_tolerance, mz_tolerance)
        print('extract time:' + str(time.time() - start_time))

        start_time = time.time()
        self.write_to_csv(bat_pc, window_mzs, window_rts)
        print('Write time:' + str(time.time() - start_time))

        print(self.output_dir + ' generated')


class DIATargetGenerator(_BasePCGenerator, _Extractor):
    def __init__(self, ms_file_path, output_dir, lib_path, rt_tolerance, mz_tolerance, min_intensity=128):
        super().__init__(ms_file_path=ms_file_path, output_dir=output_dir)
        self.rt_tolerance = rt_tolerance
        self.mz_tolerance = mz_tolerance

        self.load_csv_lib(lib_path)
        # self.load_ms1_spectra(min_intensity)
        self.load_ms2_spectra(min_intensity)

    def load_csv_lib(self, lib_path):
        lib_file = open(lib_path, 'r')
        reader = csv.reader(lib_file)
        header = next(reader)
        self.lib = {col_name: [] for col_name in header}
        for line in reader:
            for i in range(len(line)):
                self.lib[header[i]] += [line[i]]

    def generate(self):
        assert self.lib.__contains__('precursor m/z') and self.lib.__contains__('m/z')
        precursor_mzs = np.array(self.lib['precursor m/z']).astype(np.float32)
        target_mzs = self.lib['m/z']
        if self.lib.__contains__('RT(min)'):
            target_rts = np.array(self.lib['RT(min)']).astype(np.float32)
        else:
            target_rts = None

        start_time = time.time()
        bat_id = []
        bat_pc = []
        bat_mzs = []
        bat_rts = []
        for i, precursor_mz in enumerate(precursor_mzs):
            for window in self.ms2_spectra.keys():
                if (precursor_mz > window[0] + 0.1) and (precursor_mz < window[1] - 0.1):
                    break
            frag_mzs = np.array(target_mzs[i].split(',')).astype(np.float32)
            frag_rts = np.ones(len(frag_mzs)) * target_rts[i:i+1]
            bat_pc += self._bat_extract_pc(self.ms2_spectra[window], frag_mzs, frag_rts,
                                           self.rt_tolerance, self.mz_tolerance)
            bat_mzs += frag_mzs.tolist()
            bat_rts += frag_rts.tolist()
            bat_id += [self.lib['ID'][i] for n in range(len(frag_mzs))]
        print('extract time:' + str(time.time() - start_time))
        self.write_to_csv(bat_pc, bat_mzs, bat_rts, bat_id)
        print(self.output_dir + ' generated')


class DIAUntargetGenerator(_BasePCGenerator, _Extractor):
    def __init__(self, ms_file_path, output_dir):
        super().__init__(ms_file_path=ms_file_path, output_dir=output_dir)


def extract(args):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if args.data_dir.lower().endswith('.mzml'):
        data_path = [args.data_dir]
    else:
        data_path = glob.glob(os.path.join(args.data_dir, '*'))
    data_paths = ['/home/nico/workspace/python/3D-MSNet/dataset/PXD001091/mzml/130124_dilA_1_01.mzML',
                 '/home/nico/workspace/python/3D-MSNet/dataset/PXD001091/mzml/130124_dilA_1_02.mzML',
                 '/home/nico/workspace/python/3D-MSNet/dataset/PXD001091/mzml/130124_dilA_1_03.mzML',
                 '/home/nico/workspace/python/3D-MSNet/dataset/PXD001091/mzml/130124_dilA_1_04.mzML']
    for path in data_path:
        if path in data_paths:
            continue
        if not path.lower().endswith('.mzml'):
            continue
        output_folder_name = path.split('/')[-1].split('.')[0]
        if args.__contains__('lib_path') and args.lib_path is not None:
            # Target
            output_folder_name = 'Target-' + output_folder_name
            output_dir = os.path.join(args.output_dir, output_folder_name)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            print('Start to generate ' + output_folder_name)
            start_time = time.time()
            if args.mode == 'DDA':
                pc_generator = DDATargetGenerator(path, output_dir, args.lib_path, rt_tolerance=args.window_rt_width/2,
                                                  mz_tolerance=args.window_mz_width/2, min_intensity=args.min_intensity)
            else:
                pc_generator = DIATargetGenerator(path, output_dir, args.lib_path, rt_tolerance=args.window_rt_width/2,
                                                  mz_tolerance=args.window_mz_width/2, min_intensity=args.min_intensity)

            print('File reading time cost: ' + str(time.time() - start_time))
        else:
            # UnTarget
            output_folder_name = 'Untarget-' + output_folder_name
            output_dir = os.path.join(args.output_dir, output_folder_name)

            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            print('Start to generate ' + output_folder_name)
            start_time = time.time()
            pc_generator = DDAUntargetGenerator(path, output_dir,
                                                from_mz=args.from_mz, to_mz=args.to_mz,
                                                from_rt=args.from_rt, to_rt=args.to_rt,
                                                window_mz_width=args.window_mz_width,
                                                window_rt_width=args.window_rt_width,
                                                max_peak_mz_width=args.max_peak_mz_width,
                                                max_peak_rt_width=args.max_peak_rt_width,
                                                min_intensity=args.min_intensity)
            print("File reading time cost: " + str(time.time() - start_time))

        pc_generator.generate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DDA data preparation')

    parser.add_argument('--data_dir', type=str, help='converted file dir', required=True)
    parser.add_argument('--output_dir', type=str, help='point cloud output directory', required=True)
    parser.add_argument('--lib_path', type=str, help='library')
    parser.add_argument('--window_mz_width', type=float, help='window_mz_width', default=0.8)
    parser.add_argument('--window_rt_width', type=float, help='window_rt_width', default=6)
    parser.add_argument('--min_intensity', type=float, help='min_intensity', default=128)
    parser.add_argument('--from_mz', type=float, help='from_mz', default=100)
    parser.add_argument('--to_mz', type=float, help='to_mz', default=1300)
    parser.add_argument('--from_rt', type=float, help='from_rt', default=0)
    parser.add_argument('--to_rt', type=float, help='to_rt', default=40)
    parser.add_argument('--max_peak_mz_width', type=float, help='max_peak_mz_width', default=0.1)
    parser.add_argument('--max_peak_rt_width', type=float, help='max_peak_rt_width', default=0.5)
    args = parser.parse_args()

    extract(args)
