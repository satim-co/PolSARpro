#!/usr/bin/env python3

import os
import sys
import platform
import datetime
import subprocess
import time
import math
import json
sys.path.append(r'../src/')
sys.path.append(r'../src/data_process_sngl/')
sys.path.append(r'../src/speckle_filter/')


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
    if os.environ.get('GITHUB_RUN_NUMBER') is not None:
        # aws_bucket_dir = os.path.join(home, 's3-bucket')
        aws_bucket_dir = home
        DIR_PATTERN = os.path.join(home, 'polsarpro/patterns')
        DIR_IN = os.path.join(home, 'polsarpro/in/')
        DIR_OUT = os.path.join(home, f"polsarpro/out/{os.environ.get('GITHUB_RUN_NUMBER')}")
        DIR_ARTIFACTS = os.path.join(DIR_OUT, 'artifacts')
    else:
        DIR_PATTERN = os.path.join(home, 'polsarpro/pattern/')
        DIR_IN = os.path.join(home, 'polsarpro/in/')
        DIR_OUT = os.path.join(home, 'polsarpro/out/')
        DIR_BIN = os.path.join(home, 'projects/polsarpro/Soft/bin/debug')
        DIR_ARTIFACTS = os.path.join(DIR_OUT, 'artifacts')
else:
    print(f'unknown platform: {platform.system()}')
    sys.exit(1)


class Logger(object):
    timestamp = None
    log_dir = None
    log_test = None

    def __init__(self):
        Logger.timestamp = datetime.datetime.now()
        Logger.log_dir = os.path.join(DIR_ARTIFACTS, Logger.timestamp.strftime('%Y%m%dT%H%M%S'))
        Logger.make_dirs(Logger.log_dir)
        self.log_file_name = os.path.join(Logger.log_dir, Logger.timestamp.strftime('tests_log_all.txt'))
        self.log_file = open(self.log_file_name, "w")
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
        self.log_file.write(message)
        if Logger.log_test is not None:
            Logger.log_test.write(message)

    def flush(self):
        pass

    @staticmethod
    def make_dirs(path):
        if not os.path.exists(path):
            os.makedirs(path)


class Module:
    ADD_MODULE_TO_DIR_IN = False

    def __init__(self, name, lang):
        self.name = name
        self.lang = lang
        self.params = {}
        self.dir_pattern = os.path.join(DIR_PATTERN, self.name.lower())
        if Module.ADD_MODULE_TO_DIR_IN is True:
            self.dir_in = os.path.join(DIR_IN, self.name.lower())
        else:
            self.dir_in = DIR_IN
        self.dir_artifacat_module = os.path.join(Logger.log_dir, self.name.lower())
        self.dir_out = os.path.join(self.dir_artifacat_module, 'out')
        self.dir_bin = DIR_BIN
        self.skip = ''
        self.result = 'SKIP'
        time = datetime.datetime.now()
        self.stdout = ''
        self.time = time - time
        self.log_file_name = os.path.join(self.dir_artifacat_module, f'{self.name.lower()}.txt')
        self.log_file = None

    def get_log(self):
        if self.log_file is None:
            self.log_file = open(self.log_file_name, 'w')
        return self.log_file

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
        if len(files_out) != len(files_pattern):
            test_result = False
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
        Logger.make_dirs(self.dir_out)
        Logger.log_test = self.get_log()
        print(f'{"START":<10}: {self.name}')
        time_start = datetime.datetime.now()
        for i, (k, v) in enumerate(self.params.items()):
            if i == 0:
                print(f'{"PARAMS":<10}: {k}: {v}')
            else:
                print(f'{"":<10}  {k}: {v}')
        print(f'{"DIR_IN":<10}: {self.dir_in}')
        print(f'{"DIR_OUT":<10}: {self.dir_out}')
        print(f'{"PATTERN":<10}: {self.dir_pattern}')
        if self.skip != '' and self.lang == 'py':
            self.time = datetime.datetime.now() - time_start
            print(f'\n{"RESULT":<10}: {self.result} - reason: {self.skip}')
            Logger.log_test = None
            return self.time, self.result, self.skip
        Logger.make_dirs(self.dir_in)
        Logger.make_dirs(self.dir_pattern)
        if self.lang == 'py':
            m = f'../src/data_process_sngl/{self.name}.{self.lang}'
            print(f'\nimport module {m}')
            module = __import__(self.name)
            module.main(**self.params)
        elif self.lang == 'c':
            params = []
            params.append(os.path.join(self.dir_bin, f'{self.name}.exe'))
            for k, v in self.params.items():
                params.append(f'-{k}')
                params.append(f'{v}')
            proc = subprocess.Popen(params, stdout=subprocess.PIPE)
            while proc.poll() is None:
                print(proc.stdout.read1().decode('utf-8'), end='')
                time.sleep(0.5)
        self.time = datetime.datetime.now() - time_start
        info = str(self.time).split(".", 2)[0]
        if self.compare_result() is False:
            self.result = 'FAIL'
        else:
            self.result = 'PASS'
        print(f'\n{"RESULT":<10}: {self.result}')
        print(f'{"TIME":<10}: {info}')
        Logger.log_test = None
        return self.time, self.result, ''


class AriiAnned3ComponentsDecomposition(Module):
    MODULE_NAME = 'arii_anned_3components_decomposition'

    def __init__(self, skip='', lang='py'):
        super().__init__(AriiAnned3ComponentsDecomposition.MODULE_NAME, lang)
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
    MODULE_NAME = 'arii_nned_3components_decomposition'

    def __init__(self, skip='', lang='py'):
        super().__init__(AriiNned3ComponentsDecomposition.MODULE_NAME, lang)
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
    MODULE_NAME = 'freeman_2components_decomposition'

    def __init__(self, skip='', lang='py'):
        super().__init__(Freeman2ComponentsDecomposition.MODULE_NAME, lang)
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
    MODULE_NAME = ID_CLASS_GEN = 'id_class_gen'

    def __init__(self, skip='', lang='py'):
        super().__init__(IdClassGen.MODULE_NAME, lang)
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
    MODULE_NAME = 'OPCE'

    def __init__(self, skip='', lang='py'):
        super().__init__(Opce.MODULE_NAME, lang)
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
    MODULE_NAME = 'vanzyl92_3components_decomposition'

    def __init__(self, skip='', lang='py'):
        super().__init__(Vanzyl92_3ComponentsDecomposition.MODULE_NAME, lang)
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
    MODULE_NAME = 'wishart_supervised_classifier'

    def __init__(self, skip='', lang='py'):
        super().__init__(WishartSupervisedClassifier.MODULE_NAME, lang)
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
    MODULE_NAME = 'wishart_h_a_alpha_classifier'

    def __init__(self, skip='', lang='py'):
        super().__init__(WishartHAAlphaClassifier.MODULE_NAME, lang)
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


class LeeRefinedFilter(Module):
    MODULE_NAME = 'lee_refined_filter'

    def __init__(self, skip='', lang='py'):
        super().__init__(LeeRefinedFilter.MODULE_NAME, lang)
        self.skip = skip
        self.params['id'] = self.dir_in
        self.params['od'] = self.dir_out
        self.params['iodf'] = 'T3'
        self.params['nw'] = 7
        self.params['nlk'] = 7
        self.params['ofr'] = 0
        self.params['ofc'] = 0
        self.params['fnr'] = 18432
        self.params['fnc'] = 1248
        self.params['errf'] = os.path.join(self.dir_out, 'MemoryAllocError.txt')
        self.params['mask'] = os.path.join(self.dir_in, 'mask_valid_pixels.bin')


class BoxcarFilter(Module):
    MODULE_NAME = 'boxcar_filter'

    def __init__(self, skip='', lang='py'):
        super().__init__(BoxcarFilter.MODULE_NAME, lang)
        self.skip = skip
        self.params['id'] = self.dir_in
        self.params['od'] = self.dir_out
        self.params['iodf'] = 'T3'
        self.params['nwr'] = 7
        self.params['nwc'] = 7
        self.params['ofr'] = 0
        self.params['ofc'] = 0
        self.params['fnr'] = 18432
        self.params['fnc'] = 1248
        self.params['errf'] = os.path.join(self.dir_out, 'MemoryAllocError.txt')
        self.params['mask'] = os.path.join(self.dir_in, 'mask_valid_pixels.bin')


class CloudeDecomposition(Module):
    MODULE_NAME = 'cloude_decomposition'

    def __init__(self, skip='', lang='py'):
        super().__init__(CloudeDecomposition.MODULE_NAME, lang)
        self.skip = skip
        self.params['id'] = self.dir_in
        self.params['od'] = self.dir_out
        self.params['iodf'] = 'T3'
        self.params['nwr'] = 7
        self.params['nwc'] = 7
        self.params['ofr'] = 0
        self.params['ofc'] = 0
        self.params['fnr'] = 18432
        self.params['fnc'] = 1248
        self.params['errf'] = os.path.join(self.dir_out, 'MemoryAllocError.txt')
        self.params['mask'] = os.path.join(self.dir_in, 'mask_valid_pixels.bin')


class FreemanDecomposition(Module):
    MODULE_NAME = 'freeman_decomposition'

    def __init__(self, skip='', lang='py'):
        super().__init__(FreemanDecomposition.MODULE_NAME, lang)
        self.skip = skip
        self.params['id'] = self.dir_in
        self.params['od'] = self.dir_out
        self.params['iodf'] = 'T3'
        self.params['nwr'] = 7
        self.params['nwc'] = 7
        self.params['ofr'] = 0
        self.params['ofc'] = 0
        self.params['fnr'] = 18432
        self.params['fnc'] = 1248
        self.params['errf'] = os.path.join(self.dir_out, 'MemoryAllocError.txt')
        self.params['mask'] = os.path.join(self.dir_in, 'mask_valid_pixels.bin')


class HAAlphaDecomposition (Module):
    MODULE_NAME = 'h_a_alpha_decomposition'

    def __init__(self, skip='', lang='py'):
        super().__init__(HAAlphaDecomposition.MODULE_NAME, lang)
        self.skip = skip
        self.params['id'] = self.dir_in
        self.params['od'] = self.dir_out
        self.params['iodf'] = 'T3'
        self.params['nwr'] = 7
        self.params['nwc'] = 7
        self.params['ofr'] = 0
        self.params['ofc'] = 0
        self.params['fnr'] = 18432
        self.params['fnc'] = 1248
        self.params['fl1'] = 1
        self.params['fl2'] = 0
        self.params['fl3'] = 0
        self.params['fl4'] = 0
        self.params['fl5'] = 0
        self.params['fl6'] = 0
        self.params['fl7'] = 0
        self.params['fl8'] = 0
        self.params['fl9'] = 0
        self.params['errf'] = os.path.join(self.dir_out, 'MemoryAllocError.txt')
        self.params['mask'] = os.path.join(self.dir_in, 'mask_valid_pixels.bin')


class Yamaguchi3ComponentsDecomposition(Module):
    MODULE_NAME = 'yamaguchi_3components_decomposition'

    def __init__(self, skip='', lang='py'):
        super().__init__(Yamaguchi3ComponentsDecomposition.MODULE_NAME, lang)
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


class Yamaguchi4ComponentsDecomposition(Module):
    MODULE_NAME = 'yamaguchi_4components_decomposition'

    def __init__(self, skip='', lang='py'):
        super().__init__(Yamaguchi4ComponentsDecomposition.MODULE_NAME, lang)
        self.skip = skip
        self.params['id'] = self.dir_in
        self.params['od'] = self.dir_out
        self.params['iodf'] = 'T3'
        self.params['mod'] = 'Y4R'
        self.params['nwr'] = 3
        self.params['nwc'] = 3
        self.params['ofr'] = 0
        self.params['ofc'] = 0
        self.params['fnr'] = 18432
        self.params['fnc'] = 1248
        self.params['errf'] = os.path.join(self.dir_out, 'MemoryAllocError.txt')
        self.params['mask'] = os.path.join(self.dir_in, 'mask_valid_pixels.bin')


class ModuleLauncher:
    ARG_HELP = '-h'
    ARG_VERBOSE = '-v'
    ARG_ADD_MODULE_TO_DIR_IN = '--add-module-to-dir-in'

    def __init__(self):
        sys.stdout = Logger()

    def prepare_junit_result(self):
        xml_test_suites = '''<?xml version="1.0" encoding="UTF-8" ?>
<testsuites name="Test modules" tests="{}" failures="{}" errors="{}" skipped="{}" assertions="{}" time="{}" timestamp="{}">
    {}
</testsuites>'''

        xml_test_suite = '''<testsuite name="Tests.Modules" tests="{}" failures="{}" errors="{}" skipped="{}" assertions="{}" time="{}" timestamp="{}" file="{}">
       <system-out>{}</system-out>
       <system-err>{}</system-err>{}
   </testsuite>'''

        xml_test_case_skipped = '''
       <testcase name="{}" classname="Tests.Modules" assertions="{}" time="{}" file="{}" line="{}">
           <skipped message="{}" />
       </testcase>'''

        xml_test_case_failed = '''
       <testcase name="{}" classname="Tests.Modules" assertions="{}" time="{}" file="{}" line="{}">
           <system-out>{}</system-out>
           <system-err>{}</system-err>
           <failure message="{}" type="Fail">
               <!-- Failure description or stack trace -->
           </failure>
        </testcase>'''

        xml_test_case_error = '''
       <testcase name="{}" classname="Tests.Modules" assertions="{}" time="{}" file="{}" line="{}">
           <error message="{}" type="Error">
               <!-- Error description or stack trace -->
           </error>
       </testcase>'''

        xml_test_case_with_output = '''
       <testcase name="{}" classname="Tests.Modules" assertions="{}" time="{}" file="{}" line="0">
           <system-out>{}</system-out>
           <system-err>{}</system-err>
       </testcase>'''

        number_of_tests = len(self.running_modules)  # in this file
        number_of_failed_tests = len([x for x in self.running_modules if x.result == 'FAIL'])  # in this file
        number_of_errored_tests = len([x for x in self.running_modules if x.result == 'ERROR'])  # in this file
        number_of_skipped_tests = len([x for x in self.running_modules if x.result == 'SKIP'])  # in this file
        number_of_assertions = 0  # for all tests in this file
        time_aggregated = 0  # time of all tests in this file in seconds
        timestamp = Logger.timestamp.strftime("%Y%m%dT%H%M%S")  # Date and time of when the test run was executed (in ISO 8601 format)
        file = os.path.basename(sys.argv[0])
        line = 0
        junit_report_xml_tests = ''
        for c, m in enumerate(self.running_modules):
            total_seconds = m.time.total_seconds()
            time_aggregated += total_seconds
            if m.result == 'PASS':
                junit_report_xml_tests += xml_test_case_with_output.format(m.name, number_of_assertions, total_seconds, file, line, m.stdout, '')
            elif m.result == 'FAIL':
                junit_report_xml_tests += xml_test_case_failed.format(m.name, number_of_assertions, total_seconds, file, line, m.stdout, '', 'FAIL')
            elif m.result == 'SKIP':
                junit_report_xml_tests += xml_test_case_skipped.format(m.name, number_of_assertions, total_seconds, file, line, m.skip)
            elif m.result == 'ERROR':
                junit_report_xml_tests += xml_test_case_error.format(m.name, number_of_assertions, total_seconds, file, line, m.stdout)

        junit_report_xml_suite = xml_test_suite.format(number_of_tests, number_of_failed_tests, number_of_errored_tests, number_of_skipped_tests, number_of_assertions, time_aggregated, timestamp, file, '', '', junit_report_xml_tests)
        junit_report_xml_suites = xml_test_suites.format(number_of_tests, number_of_failed_tests, number_of_errored_tests, number_of_skipped_tests, number_of_assertions, time_aggregated, timestamp, junit_report_xml_suite)

        junit_report_xml = os.path.join(Logger.log_dir, 'junit_report.xml')
        with open(junit_report_xml, 'w') as f:
            f.write(junit_report_xml_suites)
        print(f'Prepare: {junit_report_xml}\n')

    def preaper_modules(self):
        self.modules = []
        self.modules.append(AriiAnned3ComponentsDecomposition)
        self.modules.append(AriiNned3ComponentsDecomposition)
        self.modules.append(Freeman2ComponentsDecomposition)
        self.modules.append(IdClassGen)
        self.modules.append(Opce)
        self.modules.append(Vanzyl92_3ComponentsDecomposition)
        self.modules.append(WishartSupervisedClassifier)
        self.modules.append(WishartHAAlphaClassifier)
        self.modules.append(LeeRefinedFilter)
        self.modules.append(BoxcarFilter)
        self.modules.append(CloudeDecomposition)
        self.modules.append(FreemanDecomposition)
        self.modules.append(HAAlphaDecomposition)
        self.modules.append(Yamaguchi3ComponentsDecomposition)
        self.modules.append(Yamaguchi4ComponentsDecomposition)

    def print_usage(self):
        print('\nUsage:')
        print(f'\t{sys.argv[0]} [module] [lang] [{ModuleLauncher.ARG_VERBOSE}] [{ModuleLauncher.ARG_ADD_MODULE_TO_DIR_IN}]')
        for m in self.modules:
            print(f'\t{sys.argv[0]} [{m.MODULE_NAME}] [py|c] [{ModuleLauncher.ARG_VERBOSE}] [{ModuleLauncher.ARG_ADD_MODULE_TO_DIR_IN}]')
        print(f'\t{sys.argv[0]} [all] [py|cpp] [{ModuleLauncher.ARG_VERBOSE}] [{ModuleLauncher.ARG_ADD_MODULE_TO_DIR_IN}]')
        print(f'\tNo arguments means: {sys.argv[0]} all py')
        print('\n')

    def run(self):
        self.preaper_modules()
        module = 'all'
        lang = 'py'
        verbose = ''
        Module.ADD_MODULE_TO_DIR_IN = False
        for i in range(1, len(sys.argv)):
            arg = sys.argv[i]
            if arg in [m.MODULE_NAME for m in self.modules]:
                module = arg
            elif arg in ['py', 'c']:
                lang = arg
            elif arg == ModuleLauncher.ARG_VERBOSE:
                verbose = 'verbose'
            elif arg == ModuleLauncher.ARG_HELP:
                self.print_usage()
                exit(0)
            elif arg == ModuleLauncher.ARG_ADD_MODULE_TO_DIR_IN:
                Module.ADD_MODULE_TO_DIR_IN = True
        print(f'module: {module} lang: {lang}')
        summary_time = []
        summary = [['Np.', 'MODULE', 'RESULT', 'INFO', 'TIME']]
        print('============================================================================================')
        print(("{: >%s}" % 45).format('-== BEGIN ==-'))
        print('--------------------------------------------------------------------------------------------\n')
        skip_modules_file = 'skip_modules.json'
        skip_modules = None
        if os.path.exists(skip_modules_file) is True:
            with open(skip_modules_file, 'r') as f:
                skip_modules = json.load(f)

        self.running_modules = []
        for c, m in enumerate(self.modules):
            instance = m(lang=lang)
            self.running_modules.append(instance)
            if m.MODULE_NAME == module or module == 'all':
                if verbose == 'verbose':
                    instance.params['v'] = None
                if skip_modules is not None:
                    for t in [v for v in skip_modules if v['name'] == instance.name]:
                        instance.skip = t['reason']
                t, r, i = instance.run()
                summary.append([c + 1, instance.name, r, i, str(t).split(".", 2)[0]])
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
        self.prepare_junit_result()


if __name__ == "__main__":
    module_launcher = ModuleLauncher()
    module_launcher.run()
