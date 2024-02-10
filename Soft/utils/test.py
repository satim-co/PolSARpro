#!/usr/bin/env python3

import os
import sys
import platform
import datetime
import subprocess
import time
import math
import logging
sys.path.append(r'../src/')
sys.path.append(r'../src/data_process_sngl/')


DIR_BIN = None
DIR_IN = None
DIR_OUT = None
DIR_PATTERN = None
DIR_ARTIFACTS = None

if platform.system().lower().startswith('win') is True:
    DIR_PATTERN = 'c:\\Projekty\\polsarpro.svn\\pattern\\'
    DIR_IN = 'c:\\Projekty\\polsarpro.svn\\in\\'
    DIR_OUT = 'c:\\Projekty\\polsarpro.svn\\out\\'
    DIR_BIN = 'c:\\Program Files (x86)\\PolSARpro_v6.0.3_Biomass_Edition\\Soft\\bin\\data_process_sngl\\'
    DIR_ARTIFACTS = os.path.join(DIR_OUT, 'artifacts')
elif platform.system().lower().startswith('lin') is True:
    home = os.environ["HOME"]
    DIR_PATTERN = os.path.join(home, 'polsarpro/pattern/')
    DIR_IN = os.path.join(home, 'polsarpro/in/')
    DIR_OUT = os.path.join(home, 'polsarpro/out/')
    DIR_BIN = os.path.join(home, 'projects/polsarpro/Soft/bin/debug')
    DIR_ARTIFACTS = os.path.join(DIR_OUT, 'artifacts')
else:
    print(f'unknown platform: {platform.system()}')
    sys.exit(1)


ARII_ANNED_3COMPONENTS_DECOMPOSITION = 'arii_anned_3components_decomposition'
ARII_NNED_3COMPONENTS_DECOMPOSITION = 'arii_nned_3components_decomposition'
FREEMAN_2COMPONENTS_DECOMPOSITION = 'freeman_2components_decomposition'
ID_CLASS_GEN = 'id_class_gen'
OPCE = 'OPCE'
VANZYL92_3COMPONENTS_DECOMPOSITION = 'vanzyl92_3components_decomposition'
WISHART_SUPERVISED_CLASSIFIER = 'wishart_supervised_classifier'
WISHART_H_A_ALPHA_CLASSIFIER = 'wishart_h_a_alpha_classifier'


class Logger(object):
    def __init__(self):
        Logger.make_dirs(DIR_ARTIFACTS)
        log_file_name = os.path.join(DIR_ARTIFACTS, datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M_%S.log'))
        self.log = open(log_file_name, "w")
        print(f'The log file: {log_file_name} is created!')
        self.terminal = sys.stdout

    @staticmethod
    def print_table(rows, footer, line_between_rows=True):
        max_col_lens = list(map(max, zip(*[(len(str(cell)) for cell in row) for row in rows])))
        print('+' + '+'.join('=' * (n + 2) for n in max_col_lens) + '+')
        rows_separator_header = '+' + '+'.join('=' * (n + 2) for n in max_col_lens) + '+'
        rows_separator_row = '+' + '+'.join('-' * (n + 2) for n in max_col_lens) + '+'
        row_fstring = ' | '.join("{: <%s}" % n for n in max_col_lens)
        for i, row in enumerate(rows):
            print('|', row_fstring.format(*map(str, row)), '|')
            if line_between_rows and i == 0:
                print(rows_separator_header)
            elif line_between_rows and i < len(rows) - 1:
                print(rows_separator_row)
        if len(footer) > 0:
            rows_separator_footer = '+' + '='.join('=' * (n + 2) for n in max_col_lens) + '+'
            print(rows_separator_footer)
            row_fstring = ''
            max_col_lens_footer = list(map(max, zip(*[(len(str(cell)) for cell in row) for row in footer])))
            tn = 0
            for i, max_col_len_data in enumerate(max_col_lens):
                max_col_len_foot = max_col_lens_footer[i]
                if i > 0:
                    row_fstring += '   '
                if max_col_len_data >= max_col_len_foot:
                    r = max_col_len_data - tn
                    if r >= 0:
                        row_fstring += "{: <%s}" % (r)
                        tn = 0
                    else:
                        row_fstring += "{: <%s}" % (0)
                        tn = math.fabs(r)
                else:
                    row_fstring += "{: <%s}" % max_col_len_foot
                    tn = max_col_len_foot - max_col_len_data
            for i, row in enumerate(footer):
                print('|', row_fstring.format(*map(str, row)), '|')
            print('+' + '='.join('=' * (n + 2) for n in max_col_lens) + '+')
        else:
            print('+' + '+'.join('-' * (n + 2) for n in max_col_lens) + '+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass 

    @staticmethod
    def make_dirs(path):
        if not os.path.exists(path):
            os.makedirs(path)
            print(f'The directory: {path} is created!')


class Module:
    LANG = 'py'

    def __init__(self, name):
        self.name = name
        self.params = {}
        self.dir_pattern = os.path.join(DIR_PATTERN, self.name.lower())
        self.dir_pattern = os.path.join(self.dir_pattern, Module.LANG)
        self.dir_in = os.path.join(DIR_IN, self.name.lower())
        self.dir_out = os.path.join(DIR_OUT, self.name.lower())
        self.dir_out = os.path.join(self.dir_out, Module.LANG)
        self.dir_bin = DIR_BIN
        self.skip = ''

    def check_md5sum(sela, file):
        params_md5sum = []
        params_md5sum.append('md5sum')
        params_md5sum.append('-b')
        params_md5sum.append(file)
        params_cut = []
        params_cut.append('cut')
        params_cut.append('-d')
        params_cut.append(' ')
        params_cut.append('-f')
        params_cut.append('1')
        p1 = subprocess.Popen(params_md5sum, stdout=subprocess.PIPE)
        p1.wait()
        p2 = subprocess.Popen(params_cut, stdin=p1.stdout, stdout=subprocess.PIPE)
        p2.wait()
        return p2.stdout.read1().decode('utf-8').strip()

    def compare_result(self):
        files_out = [os.path.join(self.dir_out, f) for f in os.listdir(self.dir_out) if os.path.isfile(os.path.join(self.dir_out, f))]
        print(f'\n\nFILES OUT:      {files_out}')
        files_pattern = [os.path.join(self.dir_pattern, f) for f in os.listdir(self.dir_pattern) if os.path.isfile(os.path.join(self.dir_pattern, f))]
        print(f'\nFILES PATTERNS: {files_pattern}')
        merged_files = zip(files_out, files_pattern)
        print('\nCOMPARE RESULTS WITH PATTERNS:')
        test_result = True
        for o, p in merged_files:
            md5sum_o = self.check_md5sum(o)
            md5sum_p = self.check_md5sum(p)
            if md5sum_o != md5sum_p:
                print(f'{o}: {md5sum_o} != {p}: {md5sum_p}')
                test_result = False
            else:
                print(f'{o}: {md5sum_o} == {p}: {md5sum_p}')
        return test_result

    def run(self):
        ext = None
        if Module.LANG == 'py':
            ext = '.py'
        elif Module.LANG == 'c':
            ext = '.c'
        print(f'{"START":<10}: {self.name}{ext}')
        result = 'SKIP'
        time_start = datetime.datetime.now()
        for i, (k, v) in enumerate(self.params.items()):
            if i == 0:
                print(f'{"PARAMS":<10}: {k}: {v}')
            else:
                print(f'{"":<10}  {k}: {v}')
        Logger.make_dirs(self.dir_pattern)
        Logger.make_dirs(self.dir_out)
        Logger.make_dirs(self.dir_in)
        if self.skip != '' and Module.LANG == 'py':
            time_finish = datetime.datetime.now() - time_start
            print(f'\n{result} {self.name:<60}: - reason: {self.skip}')
            return time_finish, result, self.skip
        if Module.LANG == 'py':
            m = f'../src/data_process_sngl/{self.name}.{Module.LANG}'
            print(f'import module {m}')
            module = __import__(self.name)
            module.main(**self.params)
        elif Module.LANG == 'c':
            params = []
            params.append(os.path.join(self.dir_bin, f'{self.name}.exe'))
            for k, v in self.params.items():
                params.append(f'-{k}')
                params.append(f'{v}')
            proc = subprocess.Popen(params, stdout=subprocess.PIPE)
            while proc.poll() is None:
                print(proc.stdout.read1().decode('utf-8'), end='')
                time.sleep(0.5)
        time_finish = datetime.datetime.now() - time_start
        info = str(time_finish).split(".", 2)[0]
        if self.compare_result() is False:
            result = 'FAIL'
        else:
            result = 'PASS'
        print(f'\n{"RESULT":<10}: {result}')
        print(f'{"TIME":<10}: {info}')
        return time_finish, result, ''


class AriiAnned3ComponentsDecomposition(Module):
    def __init__(self, skip=''):
        super().__init__(ARII_ANNED_3COMPONENTS_DECOMPOSITION)
        self.skip = skip
        self.params['id'] = self.dir_in
        self.params['od'] = self.dir_out
        self.params['iodf'] = 'T3'
        self.params['nwr'] = 3
        self.params['nwc'] = 3
        self.params['ofr'] = 0
        self.params['ofc'] = 0
        self.params['fnr'] = 18432
        self.params['fnc'] = 1248
        self.params['errf'] = os.path.join(self.dir_out, 'MemoryAllocError.txt')
        self.params['mask'] = os.path.join(self.dir_in, 'mask_valid_pixels.bin')


class AriiNned3ComponentsDecomposition(Module):
    def __init__(self, skip=''):
        super().__init__(ARII_NNED_3COMPONENTS_DECOMPOSITION)
        self.skip = skip
        self.params['id'] = self.dir_in
        self.params['od'] = self.dir_out
        self.params['iodf'] = 'T3'
        self.params['nwr'] = 3
        self.params['nwc'] = 3
        self.params['ofr'] = 0
        self.params['ofc'] = 0
        self.params['fnr'] = 18432
        self.params['fnc'] = 1248
        self.params['errf'] = os.path.join(self.dir_out, 'MemoryAllocError.txt')
        self.params['mask'] = os.path.join(self.dir_in, 'mask_valid_pixels.bin')


class Freeman2ComponentsDecomposition(Module):
    def __init__(self, skip=''):
        super().__init__(FREEMAN_2COMPONENTS_DECOMPOSITION)
        self.skip = skip
        self.params['id'] = self.dir_in
        self.params['od'] = self.dir_out
        self.params['iodf'] = 'T3'
        self.params['nwr'] = 3
        self.params['nwc'] = 3
        self.params['ofr'] = 0
        self.params['ofc'] = 0
        self.params['fnr'] = 18432
        self.params['fnc'] = 1248
        self.params['errf'] = os.path.join(self.dir_out, 'MemoryAllocError.txt')
        self.params['mask'] = os.path.join(self.dir_in, 'mask_valid_pixels.bin')


class IdClassGen(Module):
    def __init__(self, skip=''):
        super().__init__(ID_CLASS_GEN)
        self.skip = skip
        self.params['id'] = self.dir_in
        self.params['od'] = self.dir_out
        self.params['ofr'] = 0
        self.params['ofc'] = 0
        self.params['fnr'] = 18432
        self.params['fnc'] = 1248
        self.params['icf'] = os.path.join(self.dir_in, 'wishart_H_A_alpha_class_3x3.bin')
        self.params['clm'] = os.path.join(self.dir_in, 'Wishart_ColorMap16.pal')
        self.params['errf'] = os.path.join(self.dir_out, 'MemoryAllocError.txt')
        self.params['mask'] = os.path.join(self.dir_in, 'mask_valid_pixels.bin')


class Opce(Module):
    def __init__(self, skip=''):
        super().__init__(OPCE)
        self.skip = skip
        self.params['id'] = self.dir_in
        self.params['od'] = self.dir_out
        self.params['iodf'] = 'T3'
        self.params['nwr'] = 1000
        self.params['nwc'] = 1000
        self.params['ofr'] = 0
        self.params['ofc'] = 0
        self.params['fnr'] = 18432
        self.params['fnc'] = 1248
        self.params['af'] = os.path.join(self.dir_in, 'OPCE_areas.txt')
        self.params['errf'] = os.path.join(self.dir_out, 'MemoryAllocError.txt')
        self.params['mask'] = os.path.join(self.dir_in, 'mask_valid_pixels.bin')


class Vanzyl92_3ComponentsDecomposition(Module):
    def __init__(self, skip=''):
        super().__init__(VANZYL92_3COMPONENTS_DECOMPOSITION)
        self.skip = skip
        self.params['id'] = self.dir_in
        self.params['od'] = self.dir_out
        self.params['iodf'] = 'T3'
        self.params['nwr'] = 3
        self.params['nwc'] = 3
        self.params['ofr'] = 0
        self.params['ofc'] = 0
        self.params['fnr'] = 18432
        self.params['fnc'] = 1248
        self.params['errf'] = os.path.join(self.dir_out, 'MemoryAllocError.txt')
        self.params['mask'] = os.path.join(self.dir_in, 'mask_valid_pixels.bin')


class WishartSupervisedClassifier(Module):
    def __init__(self, skip=''):
        super().__init__(WISHART_SUPERVISED_CLASSIFIER)
        self.skip = skip
        self.params['id'] = self.dir_in
        self.params['od'] = self.dir_out
        self.params['iodf'] = 'T3'
        self.params['nwr'] = 3
        self.params['nwc'] = 3
        self.params['ofr'] = 0
        self.params['ofc'] = 0
        self.params['fnr'] = 18432
        self.params['fnc'] = 1248
        self.params['bmp'] = 1
        self.params['col'] = os.path.join(self.dir_in, 'Supervised_ColorMap16.pal')
        self.params['af'] = os.path.join(self.dir_in, 'wishart_training_areas_eric.txt')
        self.params['cf'] = os.path.join(self.dir_in, 'wishart_training_cluster_centers.bin')
        self.params['errf'] = os.path.join(self.dir_out, 'MemoryAllocError.txt')
        self.params['mask'] = os.path.join(self.dir_in, 'mask_valid_pixels.bin')


class WishartHAAlphaClassifier(Module):
    def __init__(self, skip=''):
        super().__init__(WISHART_H_A_ALPHA_CLASSIFIER)
        self.skip = skip
        self.params['id'] = self.dir_in
        self.params['od'] = self.dir_out
        self.params['iodf'] = 'T3'
        self.params['nwr'] = 3
        self.params['nwc'] = 3
        self.params['ofr'] = 0
        self.params['ofc'] = 0
        self.params['fnr'] = 18432
        self.params['fnc'] = 1248
        self.params['pct'] = 10
        self.params['nit'] = 10
        self.params['bmp'] = 1
        self.params['co8'] = os.path.join(self.dir_in, 'Wishart_ColorMap8.pal')
        self.params['co16'] = os.path.join(self.dir_in, 'Wishart_ColorMap16.pal')
        self.params['hf'] = os.path.join(self.dir_in, 'entropy.bin')
        self.params['af'] = os.path.join(self.dir_in, 'anisotropy.bin')
        self.params['alf'] = os.path.join(self.dir_in, 'alpha.bin')
        self.params['errf'] = os.path.join(self.dir_out, 'MemoryAllocError.txt')
        self.params['mask'] = os.path.join(self.dir_in, 'mask_valid_pixels.bin')



class ModuleLauncher:
    def __init__(self):
        sys.stdout = Logger()

    def prepare(self, lang):
        Module.LANG = lang
        self.modules = []
        self.modules.append(AriiAnned3ComponentsDecomposition()) # 'long processing time'))
        self.modules.append(AriiNned3ComponentsDecomposition())
        self.modules.append(Freeman2ComponentsDecomposition())
        self.modules.append(IdClassGen())
        self.modules.append(Opce())
        self.modules.append(Vanzyl92_3ComponentsDecomposition())
        self.modules.append(WishartSupervisedClassifier())
        self.modules.append(WishartHAAlphaClassifier())

    def print_usage(self):
        print('\nUsage:')
        self.prepare('py')
        for m in self.modules:
            print(f'\t{sys.argv[0]} {m.name} py|c [verbose]')
        print(f'\t{sys.argv[0]} all py|cpp [verbose]')
        print('\n')


    def run(self):
        if len(sys.argv) < 3:
            self.print_usage()
            return
        arg_module = sys.argv[1]
        arg_lang = sys.argv[2]
        if arg_lang not in ['py', 'c']:
            self.print_usage()
            return
        arg_verbose = None
        if len(sys.argv) > 3:
            arg_verbose = sys.argv[3]
        print(arg_verbose)
        summary_info = []
        summary_time = []
        self.prepare(arg_lang)
        summary = [['Np.', 'MODULE', 'RESULT', 'INFO', 'TIME']]
        print('============================================================================================')
        print(("{: >%s}" % 45).format('-== BEGIN ==-'))
        print('--------------------------------------------------------------------------------------------\n')
        for c, m in enumerate(self.modules):
            if arg_verbose == 'verbose':
                m.params['v'] = None
            if m.name == arg_module or arg_module == 'all':
                t, r, i = m.run()
                summary.append([c + 1, m.name, r, i, str(t).split(".", 2)[0]])
                summary_time.append(t)
                print('\n--------------------------------------------------------------------------------------------')
        print(("{: >%s}" % 44).format('-== END ==-'))
        print('============================================================================================')
        total_time = str(sum(summary_time, datetime.timedelta())).split('.', 2)[0]
        footer = []
        footer.append(['TOTAL TIME', '', '', '', str(total_time).split(".", 2)[0]])
        print('\nSUMMARY BEGIN\n')
        Logger.print_table(summary, footer)
        print('\nSUMMARY END\n')


if __name__ == "__main__":
    module_launcher = ModuleLauncher()
    module_launcher.run()
