"""
Polsarpro
===
util
"""
# %% [codecell] import
import errno
import os
import sys
import resource
import argparse
import datetime
import math
import platform
import logging
from pathlib import Path
import numba
import timeit


class EnterExitLog():
    def __init__(self, funcName):
        self.funcName = funcName

    def __enter__(self):
        logging.info('--= Started: %s =--' % self.funcName)
        self.init_time = datetime.datetime.now()
        return self

    def __exit__(self, type, value, tb):
        logging.info('--= Finished: %s in: %s sec =--' % (self.funcName, datetime.datetime.now() - self.init_time))


def enter_exit_func_decorator(func):
    def func_wrapper(*args, **kwargs):
        with EnterExitLog(func.__name__):
            return func(*args, **kwargs)

    return func_wrapper


class CodeTimer:
    def __init__(self, name=None):
        self.name = " '" + name + "'" if name else ''

    def __enter__(self):
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = (timeit.default_timer() - self.start) * 1000.0
        print('Code block' + self.name + ' took: ' + str(self.took) + ' ms')


class Termination:
    # Successful exit
    @staticmethod
    def success():
        sys.exit(0)

    # Abnormal termination
    @staticmethod
    def failure():
        sys.exit(1)


class RecursionAndStackLimit:
    def __init__(self, recursion_limit, stack_size):
        self.current_recursion_limit = recursion_limit
        self.default_recursion_limit = sys.getrecursionlimit()
        self.current_resource_limit = stack_size
        self.default_resource_limit = resource.getrlimit(resource.RLIMIT_STACK)

    def __enter__(self):
        sys.setrecursionlimit(self.current_recursion_limit)
        resource.setrlimit(resource.RLIMIT_STACK, self.current_resource_limit)
        logging.info(f'setrecursionlimit: {self.current_recursion_limit}, setrlimit: {self.current_resource_limit}')

    def __exit__(self, type, value, traceback):
        sys.setrecursionlimit(self.default_recursion_limit)
        resource.setrlimit(resource.RLIMIT_STACK, self.default_resource_limit)
        logging.info(f'setrecursionlimit: {self.default_recursion_limit}, setrlimit: {self.default_resource_limit}')


class ParseArgs:
    """
    Parse command line arguments.
    """

    @staticmethod
    def get_args(*args, **kwargs):
        local_args = None
        if len(args) > 0:
            local_args = args[0]
        else:
            local_args = []
            for k, v in kwargs.items():
                local_args.append(f'-{k}')
                if v is not None:
                    local_args.append(f'{v}')
        return local_args

    def __init__(self, args, desc, pol_types):
        self.parser = argparse.ArgumentParser(description=desc, add_help=False)
        self.args = args
        self.required = self.parser.add_argument_group('Required parameters')
        self.optional = self.parser.add_argument_group('Optional parameters')
        self.pol_types = pol_types

    def add_req_arg(self, n, t, h, c=None):
        self.required.add_argument(n, type=t, required=True, help=h, choices=c)

    def rem_req_arg(self, arg):
        for action in self.parser._actions:
            opts = action.option_strings
            if (opts and opts[0] == arg) or action.dest == arg:
                self.parser._remove_action(action)
                break

        for action in self.parser._action_groups:
            for group_action in action._group_actions:
                opts = group_action.option_strings
                if (opts and opts[0] == arg) or group_action.dest == arg:
                    action._group_actions.remove(group_action)
                    return

    def add_opt_arg(self, n, t, h, a):
        self.optional.add_argument(n, type=t, required=False, help=h, action=a)

    def make_def_args(self):
        self.required.add_argument('-id', type=str, required=True, help='input directory')
        self.required.add_argument('-od', type=str, required=True, help='output directory')
        self.required.add_argument('-iodf', type=str, required=True, choices=self.pol_types, help='input-output data format')
        self.required.add_argument('-nwr', type=int, required=True, help='Nwin Row')
        self.required.add_argument('-nwc', type=int, required=True, help='Nwin Col')
        self.required.add_argument('-ofr', type=int, required=True, help='Offset Row')
        self.required.add_argument('-ofc', type=int, required=True, help='Offset Col')
        self.required.add_argument('-fnr', type=int, required=True, help='Final Number of Row')
        self.required.add_argument('-fnc', type=int, required=True, help='Final Number of Col')

        self.optional.add_argument('-mask', type=str, required=False, help='mask file (valid pixels)')
        self.optional.add_argument('-errf', type=str, required=False, help='memory error file')
        self.optional.add_argument('-data', type=int, required=False, help='displays the help concerning Data Format parameter')

    def parse_args(self):
        self.optional.add_argument('-h', '--help', action='help', help='show this help message and exit')
        self.optional.add_argument('-v', help='verbose', action='store_true')
        self.optional.add_argument('-V', help='version', action="version", version=" version: 67")
        return self.parser.parse_args(self.args)

    def print_help(self):
        self.parser.print_help(sys.stderr)


class Application:
    INIT_MINMAX = 1.E+30
    EPS = 1.E-30
    FILE_PATH_LENGTH = 8192
    PI = 3.14159265358979323846

    def __init__(self, args):
        self.args = args
        if self.args.v:
            logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s %(message)s [%(filename)s:%(lineno)d]', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
            logging.getLogger('numba').setLevel(logging.WARNING)

    def check_dir(self, path):
        correct_path = Path(path)
        norm_path = os.path.normpath(correct_path)
        if not os.path.exists(norm_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), f'{norm_path}')
        return norm_path

    def check_file(self, path):
        '''
        Check file exist.
        '''
        correct_path = Path(path)
        norm_path = os.path.normpath(correct_path)
        if not os.path.exists(norm_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), f'{norm_path}')
        return norm_path

    def open_file(self, file_name, mode):
        '''
        Open input file and return the file object.
        '''
        open_file = None
        try:
            open_file = open(file_name, mode)
        except IOError:
            print("Could not open input file: ", file_name)
            raise
        return open_file

    def open_input_files(self, file_name_in, file_valid, in_datafile, n_polar_in, in_valid):
        '''
        Open input files and return the file objects and a flag indicating if the 'valid' file is present.
        '''
        flag_valid = False
        for n_pol in range(n_polar_in):
            try:
                in_datafile.append(open(file_name_in[n_pol], "rb"))
            except IOError:
                print("Could not open input file : ", file_name_in[n_pol])
                raise

        if file_valid:
            flag_valid = True
            try:
                in_valid = open(file_valid, "rb")
            except IOError:
                print("Could not open input file: ", file_valid)
                raise
        return in_datafile, in_valid, flag_valid

    def open_output_file(self, file_name_out, mode='wb'):
        '''
        Open output files and return the file objects.
        '''
        try:
            out_datafile = open(file_name_out, mode)
        except IOError:
            print(f"Could not open output file : {file_name_out}")
            raise
        return out_datafile

    def check_free_memory(self):
        if platform.system().lower().startswith('win') is True:
            output = os.popen("wmic OS get FreePhysicalMemory")
            value = output.read().split("\n")[2]
            MemoryValue = math.floor(0.75 * int(value) / 1024.)
            return MemoryValue
        elif platform.system().lower().startswith('lin') is True:
            import re
            r = re.compile('[ \t\n\r:]+')
            MemoryValue = 0
            MemoryFree = 0
            MemoryBuffer = 0
            MemoryCache = 0
            flag = 0
            with open('/proc/meminfo', 'r') as meminfo:
                lines = meminfo.readlines()
                for line in lines:
                    s = r.split(line)
                    if s[0] == 'MemFree':
                        MemoryFree = int(s[1])
                        flag = flag + 1
                    elif s[0] == 'Buffers':
                        MemoryBuffer = int(s[1])
                        flag = flag + 10
                    elif s[0] == 'Cached':
                        MemoryCache = int(s[1])
                        flag = flag + 100
                    if flag == 111:
                        logging.info(f'{MemoryFree=} {MemoryBuffer=} {MemoryCache=}')
                        MemoryValue = math.floor(0.75 * (MemoryFree + MemoryBuffer + MemoryCache) / 1024.)
                        return MemoryValue
        return 1000

    def memory_alloc(self, filememerr, Nlig, Nwin, NbBlock, NligBlock, NBlockA, NBlockB, MemAlloc):
        '''
        Description : BlockSize and Number of Blocks determination
        '''
        logging.info(locals())
        NligBlockSizeFlt = 250000. * MemAlloc
        NligBlockSizeFlt = NligBlockSizeFlt - NBlockB
        NligBlockSizeFlt = NligBlockSizeFlt / NBlockA
        NligBlockSize = math.floor(NligBlockSizeFlt)
        logging.info(f'{NligBlockSizeFlt=}')
        logging.info(f'{NligBlockSize=}')

        if NligBlockSize <= 1:
            try:
                with open(filememerr, "w") as f:
                    f.write('THE AVALAIBLE PROCESSING MEMORY')
                    f.write(f'MUST BE HIGHER THAN {MemAlloc} Mb')
                    raise ("ERROR : NOT ENOUGH MEMORY SPACE", "")
            except FileNotFoundError:
                logging.error('NOT ENOUGH MEMORY SPACE')
                logging.error('THE AVALAIBLE PROCESSING MEMORY')
                logging.error('MUST BE HIGHER THAN {MemAlloc} Mb')
                logging.error(f'{NligBlockSize=}')
                raise f'Could not open configuration file : {filememerr}'

        if NligBlockSize >= Nlig:
            NbBlock = 1
            NligBlock[0] = Nlig
        else:
            NbBlock = (int)(math.floor(Nlig / NligBlockSize))
            NligBlockRem = Nlig - NbBlock * NligBlockSize
            for ii in range(NbBlock):
                NligBlock[ii] = NligBlockSize
            if NligBlockRem < Nwin:
                NligBlock[0] += NligBlockRem
            else:
                NligBlock[NbBlock] = NligBlockRem
                NbBlock += 1
        return NbBlock

    def set_valid_pixels(self, flag_valid, n_lig_block, sub_n_col, n_win_c, n_win_l):
        '''
        Set the valid pixels for the boxcar filter based on the provided parameters.
        '''
        if not flag_valid and self.valid is not None:
            self.valid[:n_lig_block[0] + n_win_l, :sub_n_col + n_win_c] = 1.0

    def rewind(self, f):
        f.seek(0)

    def allocate_matrices(self, n_col, n_polar_out, n_win_l, n_win_c, n_lig_block, sub_n_col):
        '''
        Allocate matrices with given dimensions
        '''
        raise NotImplementedError('Fix me - please Implement this method')

    def run(self):
        '''
        Main Function for the freeman_2components_polarimetric decomposition
        Parses the input arguments, reads the input files, and processes the data using boxcar filtering.
        '''
        raise NotImplementedError('Fix me - please Implement this method')


def my_isfinite(x):
    return x == x and (x - x == 0.0) if True else False


# /* S2 matrix */
S11 = 0
S12 = 1
S21 = 2
S22 = 3

# /* IPP Full */
I411 = 0
I412 = 1
I421 = 2
I422 = 3

# /* IPP pp4 */
I311 = 0
I312 = 1
I322 = 2

# /* IPP pp5-pp6-pp7 */
I211 = 0
I212 = 1

# /* C2 matrix */
C211 = 0
C212_RE = 1
C212_IM = 2
C222 = 3

# /* C3 matrix */
C311 = 0
C312_RE = 1
C312_IM = 2
C313_RE = 3
C313_IM = 4
C322 = 5
C323_RE = 6
C323_IM = 7
C333 = 8

# /* C4 matrix */
C411 = 0
C412_RE = 1
C412_IM = 2
C413_RE = 3
C413_IM = 4
C414_RE = 5
C414_IM = 6
C422 = 7
C423_RE = 8
C423_IM = 9
C424_RE = 10
C424_IM = 11
C433 = 12
C434_RE = 13
C434_IM = 14
C444 = 15

# /* T2 matrix */
T211 = 0
T212_RE = 1
T212_IM = 2
T222 = 3

# /* T3 matrix */
T311 = 0
T312_RE = 1
T312_IM = 2
T313_RE = 3
T313_IM = 4
T322 = 5
T323_RE = 6
T323_IM = 7
T333 = 8

# /* T4 matrix */
T411 = 0
T412_RE = 1
T412_IM = 2
T413_RE = 3
T413_IM = 4
T414_RE = 5
T414_IM = 6
T422 = 7
T423_RE = 8
T423_IM = 9
T424_RE = 10
T424_IM = 11
T433 = 12
T434_RE = 13
T434_IM = 14
T444 = 15

# /* C2 or T2 matrix */
X211 = 0
X212_RE = 1
X212_IM = 2
X222 = 3
X212 = 4

# /* C3 or T3 matrix */
X311 = 0
X312_RE = 1
X312_IM = 2
X313_RE = 3
X313_IM = 4
X322 = 5
X323_RE = 6
X323_IM = 7
X333 = 8
X312 = 9
X313 = 10
X323 = 11

# /* C4 or T4 matrix */
X411 = 0
X412_RE = 1
X412_IM = 2
X413_RE = 3
X413_IM = 4
X414_RE = 5
X414_IM = 6
X422 = 7
X423_RE = 8
X423_IM = 9
X424_RE = 10
X424_IM = 11
X433 = 12
X434_RE = 13
X434_IM = 14
X444 = 15
X412 = 16
X413 = 17
X414 = 18
X423 = 19
X424 = 20
X434 = 21

# /* T6 matrix */
T611 = 0
T612_RE = 1
T612_IM = 2
T613_RE = 3
T613_IM = 4
T614_RE = 5
T614_IM = 6
T615_RE = 7
T615_IM = 8
T616_RE = 9
T616_IM = 10
T622 = 11
T623_RE = 12
T623_IM = 13
T624_RE = 14
T624_IM = 15
T625_RE = 16
T625_IM = 17
T626_RE = 18
T626_IM = 19
T633 = 20
T634_RE = 21
T634_IM = 22
T635_RE = 23
T635_IM = 24
T636_RE = 25
T636_IM = 26
T644 = 27
T645_RE = 28
T645_IM = 29
T646_RE = 30
T646_IM = 31
T655 = 32
T656_RE = 33
T656_IM = 34
T666 = 35


# %% [codecell] pol_type_config
def pol_type_config(pol_type):
    """Check the polarimetric format configuration"""
    config = False
    pol_type_config = [
        "C2",
        "C2T2",
        "C3",
        "C3T3",
        "C4",
        "C4T4",
        "C4C3",
        "C4T3",
        "T2",
        "T2C2",
        "T3",
        "T3C3",
        "T4",
        "T4C4",
        "T4C3",
        "T4T3",
        "T6",
        "S2SPPpp1",
        "S2SPPpp2",
        "S2SPPpp3",
        "S2IPPpp4",
        "S2IPPpp5",
        "S2IPPpp6",
        "S2IPPpp7",
        "S2IPPfull",
        "S2",
        "S2C3",
        "S2C4",
        "S2T3",
        "S2T4",
        "S2T6",
        "SPP",
        "SPPC2",
        "SPPT2",
        "SPPT4",
        "SPPIPP",
        "IPP",
        "Ixx",
        "S2C2pp1",
        "S2C2pp2",
        "S2C2pp3",
        "S2SPPlhv",
        "S2SPPrhv",
        "S2SPPpi4",
        "S2C2lhv",
        "S2C2rhv",
        "S2C2pi4",
        "C2IPPpp5",
        "C2IPPpp6",
        "C2IPPpp7",
        "C3C2pp1",
        "C3C2pp2",
        "C3C2pp3",
        "C3C2lhv",
        "C3C2rhv",
        "C3C2pi4",
        "C3IPPpp4",
        "C3IPPpp5",
        "C3IPPpp6",
        "C3IPPpp7",
        "C4C2pp1",
        "C4C2pp2",
        "C4C2pp3",
        "C4C2lhv",
        "C4C2rhv",
        "C4C2pi4",
        "C4IPPpp4",
        "C4IPPpp5",
        "C4IPPpp6",
        "C4IPPpp7",
        "C4IPPfull",
        "T3C2pp1",
        "T3C2pp2",
        "T3C2pp3",
        "T3C2lhv",
        "T3C2rhv",
        "T3C2pi4",
        "T3IPPpp4",
        "T3IPPpp5",
        "T3IPPpp6",
        "T3IPPpp7",
        "T4C2pp1",
        "T4C2pp2",
        "T4C2pp3",
        "T4C2lhv",
        "T4C2rhv",
        "T4C2pi4",
        "T4IPPpp4",
        "T4IPPpp5",
        "T4IPPpp6",
        "T4IPPpp7",
        "T4IPPfull",
    ]

    if pol_type in pol_type_config:
        config = True

    # for ii in range(87):
    #     if PolTypeConfig[ii] == PolType:
    #         break

    if not config:
        raise ValueError("A processing error occurred!\nWrong Input / Output Polarimetric Data Format\nUsageHelpDataFormat\n")

    pol_type_tmp = pol_type

    if pol_type_tmp == "C2":
        n_polar_in = 4
        n_polar_out = 4
        pol_type_in = "C2"
        pol_type_out = "C2"

    elif pol_type_tmp == "C2T2":
        n_polar_in = 4
        n_polar_out = 4
        pol_type = "C2"
        pol_type_in = "C2"
        pol_type_out = "T2"

    elif pol_type_tmp == "C2IPPpp5":
        n_polar_in = 4
        n_polar_out = 2
        pol_type = "C2"
        pol_type_in = "C2"
        pol_type_out = "IPPpp5"

    elif pol_type_tmp == "C2IPPpp6":
        n_polar_in = 4
        n_polar_out = 2
        pol_type_tmp = "C2"
        pol_type_in = "C2"
        pol_type_out = "IPPpp6"

    elif pol_type_tmp == "C2IPPpp7":
        n_polar_in = 4
        n_polar_out = 2
        pol_type = "C2"
        pol_type_in = "C2"
        pol_type_out = "IPPpp7"

    elif pol_type_tmp == "C3":
        n_polar_in = 9
        n_polar_out = 9
        pol_type_in = "C3"
        pol_type_out = "C3"

    elif pol_type_tmp == "C3T3":
        n_polar_in = 9
        n_polar_out = 9
        pol_type = "C3"
        pol_type_in = "C3"
        pol_type_out = "T3"

    elif pol_type_tmp == "C3C2pp1":
        n_polar_in = 9
        n_polar_out = 4
        pol_type = "C3"
        pol_type_in = "C3"
        pol_type_out = "C2pp1"

    elif pol_type_tmp == "C3C2pp2":
        n_polar_in = 9
        n_polar_out = 4
        pol_type = "C3"
        pol_type_in = "C3"
        pol_type_out = "C2pp2"

    elif pol_type_tmp == "C3C2pp3":
        n_polar_in = 9
        n_polar_out = 4
        pol_type = "C3"
        pol_type_in = "C3"
        pol_type_out = "C2pp3"

    elif pol_type_tmp == "C3C2lhv":
        n_polar_in = 9
        n_polar_out = 4
        pol_type = "C3"
        pol_type_in = "C3"
        pol_type_out = "C2lhv"

    elif pol_type_tmp == "C3C2rhv":
        n_polar_in = 9
        n_polar_out = 4
        pol_type = "C3"
        pol_type_in = "C3"
        pol_type_out = "C2rhv"

    elif pol_type_tmp == "C3C2pi4":
        n_polar_in = 9
        n_polar_out = 4
        pol_type = "C3"
        pol_type_in = "C3"
        pol_type_out = "C2pi4"

    elif pol_type_tmp == "C3IPPpp4":
        n_polar_in = 9
        n_polar_out = 3
        pol_type = "C3"
        pol_type_in = "C3"
        pol_type_out = "IPPpp4"

    elif pol_type_tmp == "C3IPPpp5":
        n_polar_in = 9
        n_polar_out = 2
        pol_type = "C3"
        pol_type_in = "C3"
        pol_type_out = "IPPpp5"

    elif pol_type_tmp == "C3IPPpp6":
        n_polar_in = 9
        n_polar_out = 2
        pol_type = "C3"
        pol_type_in = "C3"
        pol_type_out = "IPPpp6"

    elif pol_type_tmp == "C3IPPpp7":
        n_polar_in = 9
        n_polar_out = 2
        pol_type = "C3"
        pol_type_in = "C3"
        pol_type_out = "IPPpp7"

    elif pol_type_tmp == "C4":
        n_polar_in = 16
        n_polar_out = 16
        pol_type_in = "C4"
        pol_type_out = "C4"

    elif pol_type_tmp == "C4T4":
        n_polar_in = 16
        n_polar_out = 16
        pol_type = "C4"
        pol_type_in = "C4"
        pol_type_out = "T4"

    elif pol_type_tmp == "C4C3":
        n_polar_in = 16
        n_polar_out = 9
        pol_type = "C4"
        pol_type_in = "C4"
        pol_type_out = "C3"

    elif pol_type_tmp == "C4T3":
        n_polar_in = 16
        n_polar_out = 9
        pol_type = "C4"
        pol_type_in = "C4"
        pol_type_out = "T3"

    elif pol_type_tmp == "C4C2pp1":
        n_polar_in = 16
        n_polar_out = 4
        pol_type = "C4"
        pol_type_in = "C4"
        pol_type_out = "C2pp1"

    elif pol_type_tmp == "C4C2pp2":
        n_polar_in = 16
        n_polar_out = 4
        pol_type = "C4"
        pol_type_in = "C4"
        pol_type_out = "C2pp2"

    elif pol_type_tmp == "C4C2pp3":
        n_polar_in = 16
        n_polar_out = 4
        pol_type = "C4"
        pol_type_in = "C4"
        pol_type_out = "C2pp3"

    elif pol_type_tmp == "C4C2lhv":
        n_polar_in = 16
        n_polar_out = 4
        pol_type = "C4"
        pol_type_in = "C4"
        pol_type_out = "C2lhv"

    elif pol_type_tmp == "C4C2rhv":
        n_polar_in = 16
        n_polar_out = 4
        pol_type = "C4"
        pol_type_in = "C4"
        pol_type_out = "C2rhv"

    elif pol_type_tmp == "C4C2pi4":
        n_polar_in = 16
        n_polar_out = 4
        pol_type = "C4"
        pol_type_in = "C4"
        pol_type_out = "C2pi4"

    elif pol_type_tmp == "C4IPPpp4":
        n_polar_in = 16
        n_polar_out = 3
        pol_type = "C4"
        pol_type_in = "C4"
        pol_type_out = "IPPpp4"

    elif pol_type_tmp == "C4IPPpp5":
        n_polar_in = 16
        n_polar_out = 2
        pol_type = "C4"
        pol_type_in = "C4"
        pol_type_out = "IPPpp5"

    elif pol_type_tmp == "C4IPPpp6":
        n_polar_in = 16
        n_polar_out = 2
        pol_type = "C4"
        pol_type_in = "C4"
        pol_type_out = "IPPpp6"

    elif pol_type_tmp == "C4IPPpp7":
        n_polar_in = 16
        n_polar_out = 2
        pol_type = "C4"
        pol_type_in = "C4"
        pol_type_out = "IPPpp7"

    elif pol_type_tmp == "C4IPPfull":
        n_polar_in = 16
        n_polar_out = 4
        pol_type = "C4"
        pol_type_in = "C4"
        pol_type_out = "IPPfull"

    elif pol_type_tmp == "T2":
        n_polar_in = 4
        n_polar_out = 4
        pol_type_in = "T2"
        pol_type_out = "T2"

    elif pol_type_tmp == "T2C2":
        n_polar_in = 4
        n_polar_out = 4
        pol_type = "T2"
        pol_type_in = "T2"
        pol_type_out = "C2"

    elif pol_type_tmp == "T3":
        n_polar_in = 9
        n_polar_out = 9
        pol_type_in = "T3"
        pol_type_out = "T3"

    elif pol_type_tmp == "T3C3":
        n_polar_in = 9
        n_polar_out = 9
        pol_type = "T3"
        pol_type_in = "T3"
        pol_type_out = "C3"

    elif pol_type_tmp == "T3C2pp1":
        n_polar_in = 9
        n_polar_out = 4
        pol_type = "T3"
        pol_type_in = "T3"
        pol_type_out = "C2pp1"

    elif pol_type_tmp == "T3C2pp2":
        n_polar_in = 9
        n_polar_out = 4
        pol_type = "T3"
        pol_type_in = "T3"
        pol_type_out = "C2pp2"

    elif pol_type_tmp == "T3C2pp3":
        n_polar_in = 9
        n_polar_out = 4
        pol_type = "T3"
        pol_type_in = "T3"
        pol_type_out = "C2pp3"

    elif pol_type_tmp == "T3C2lhv":
        n_polar_in = 9
        n_polar_out = 4
        pol_type = "T3"
        pol_type_in = "T3"
        pol_type_out = "C2lhv"

    elif pol_type_tmp == "T3C2rhv":
        n_polar_in = 9
        n_polar_out = 4
        pol_type = "T3"
        pol_type_in = "T3"
        pol_type_out = "C2rhv"

    elif pol_type_tmp == "T3C2pi4":
        n_polar_in = 9
        n_polar_out = 4
        pol_type = "T3"
        pol_type_in = "T3"
        pol_type_out = "C2pi4"

    elif pol_type_tmp == "T3IPPpp4":
        n_polar_in = 9
        n_polar_out = 3
        pol_type = "T3"
        pol_type_in = "T3"
        pol_type_out = "IPPpp4"

    elif pol_type_tmp == "T3IPPpp5":
        n_polar_in = 9
        n_polar_out = 2
        pol_type = "T3"
        pol_type_in = "T3"
        pol_type_out = "IPPpp5"

    elif pol_type_tmp == "T3IPPpp6":
        n_polar_in = 9
        n_polar_out = 2
        pol_type = "T3"
        pol_type_in = "T3"
        pol_type_out = "IPPpp6"

    elif pol_type_tmp == "T3IPPpp7":
        n_polar_in = 9
        n_polar_out = 2
        pol_type = "T3"
        pol_type_in = "T3"
        pol_type_out = "IPPpp7"

    elif pol_type_tmp == "T4":
        n_polar_in = 16
        n_polar_out = 16
        pol_type_in = "T4"
        pol_type_out = "T4"

    elif pol_type_tmp == "T4C4":
        n_polar_in = 16
        n_polar_out = 16
        pol_type = "T4"
        pol_type_in = "T4"
        pol_type_out = "C4"

    elif pol_type_tmp == "T4C3":
        n_polar_in = 16
        n_polar_out = 9
        pol_type = "T4"
        pol_type_in = "T4"
        pol_type_out = "C3"

    elif pol_type_tmp == "T4T3":
        n_polar_in = 16
        n_polar_out = 9
        pol_type = "T4"
        pol_type_in = "T4"
        pol_type_out = "T3"

    elif pol_type_tmp == "T4C2pp1":
        n_polar_in = 16
        n_polar_out = 4
        pol_type = "T4"
        pol_type_in = "T4"
        pol_type_out = "C2pp1"

    elif pol_type_tmp == "T4C2pp2":
        n_polar_in = 16
        n_polar_out = 4
        pol_type = "T4"
        pol_type_in = "T4"
        pol_type_out = "C2pp2"

    elif pol_type_tmp == "T4C2pp3":
        n_polar_in = 16
        n_polar_out = 4
        pol_type = "T4"
        pol_type_in = "T4"
        pol_type_out = "C2pp3"

    elif pol_type_tmp == "T4C2lhv":
        n_polar_in = 16
        n_polar_out = 4
        pol_type = "T4"
        pol_type_in = "T4"
        pol_type_out = "C2lhv"

    elif pol_type_tmp == "T4C2rhv":
        n_polar_in = 16
        n_polar_out = 4
        pol_type = "T4"
        pol_type_in = "T4"
        pol_type_out = "C2rhv"

    elif pol_type_tmp == "T4C2pi4":
        n_polar_in = 16
        n_polar_out = 4
        pol_type = "T4"
        pol_type_in = "T4"
        pol_type_out = "C2pi4"

    elif pol_type_tmp == "T4IPPpp4":
        n_polar_in = 16
        n_polar_out = 3
        pol_type = "T4"
        pol_type_in = "T4"
        pol_type_out = "IPPpp4"

    elif pol_type_tmp == "T4IPPpp5":
        n_polar_in = 16
        n_polar_out = 2
        pol_type = "T4"
        pol_type_in = "T4"
        pol_type_out = "IPPpp5"

    elif pol_type_tmp == "T4IPPpp6":
        n_polar_in = 16
        n_polar_out = 2
        pol_type = "T4"
        pol_type_in = "T4"
        pol_type_out = "IPPpp6"

    elif pol_type_tmp == "T4IPPpp7":
        n_polar_in = 16
        n_polar_out = 2
        pol_type = "T4"
        pol_type_in = "T4"
        pol_type_out = "IPPpp7"

    elif pol_type_tmp == "T4IPPfull":
        n_polar_in = 16
        n_polar_out = 4
        pol_type = "T4"
        pol_type_in = "T4"
        pol_type_out = "IPPfull"

    elif pol_type_tmp == "T6":
        n_polar_in = 36
        n_polar_out = 36
        pol_type_in = "T6"
        pol_type_out = "T6"

    elif pol_type_tmp == "S2SPPpp1":
        n_polar_in = 4
        n_polar_out = 2
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "SPPpp1"

    elif pol_type_tmp == "S2SPPpp2":
        n_polar_in = 4
        n_polar_out = 2
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "SPPpp2"

    elif pol_type_tmp == "S2SPPpp3":
        n_polar_in = 4
        n_polar_out = 2
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "SPPpp3"

    elif pol_type_tmp == "S2C2pp1":
        n_polar_in = 4
        n_polar_out = 4
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "C2pp1"

    elif pol_type_tmp == "S2C2pp2":
        n_polar_in = 4
        n_polar_out = 4
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "C2pp2"

    elif pol_type_tmp == "S2C2pp3":
        n_polar_in = 4
        n_polar_out = 4
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "C2pp3"

    elif pol_type_tmp == "S2SPPlhv":
        n_polar_in = 4
        n_polar_out = 2
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "SPPlhv"

    elif pol_type_tmp == "S2SPPrhv":
        n_polar_in = 4
        n_polar_out = 2
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "SPPrhv"

    elif pol_type_tmp == "S2SPPpi4":
        n_polar_in = 4
        n_polar_out = 2
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "SPPpi4"

    elif pol_type_tmp == "S2C2lhv":
        n_polar_in = 4
        n_polar_out = 4
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "C2lhv"

    elif pol_type_tmp == "S2C2rhv":
        n_polar_in = 4
        n_polar_out = 4
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "C2rhv"

    elif pol_type_tmp == "S2C2pi4":
        n_polar_in = 4
        n_polar_out = 4
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "C2pi4"

    elif pol_type_tmp == "S2IPPpp4":
        n_polar_in = 4
        n_polar_out = 3
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "IPPpp4"

    elif pol_type_tmp == "S2IPPpp5":
        n_polar_in = 4
        n_polar_out = 2
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "IPPpp5"

    elif pol_type_tmp == "S2IPPpp6":
        n_polar_in = 4
        n_polar_out = 2
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "IPPpp6"

    elif pol_type_tmp == "S2IPPpp7":
        n_polar_in = 4
        n_polar_out = 2
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "IPPpp7"

    elif pol_type_tmp == "S2IPPfull":
        n_polar_in = 4
        n_polar_out = 4
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "IPPfull"

    elif pol_type_tmp == "S2":
        n_polar_in = 4
        n_polar_out = 4
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "S2"

    elif pol_type_tmp == "S2C3":
        n_polar_in = 4
        n_polar_out = 9
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "C3"

    elif pol_type_tmp == "S2C4":
        n_polar_in = 4
        n_polar_out = 16
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "C4"

    elif pol_type_tmp == "S2T3":
        n_polar_in = 4
        n_polar_out = 9
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "T3"

    elif pol_type_tmp == "S2T4":
        n_polar_in = 4
        n_polar_out = 16
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "T4"

    elif pol_type_tmp == "S2T6":
        n_polar_in = 4
        n_polar_out = 36
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "T6"

    # elif pol_type_tmp == "SPP":
    #     n_polar_in = 2
    #     n_polar_out = 2
    #     pol_type = "SPP"
    #     pol_type_in = "SPP" + polar_type
    #     pol_type_out = "SPP" + polar_type

    # elif pol_type_tmp == "SPPC2":
    #     n_polar_in = 2
    #     n_polar_out = 4
    #     pol_type = "SPP"
    #     pol_type_in = "SPP" + polar_type
    #     pol_type_out = "C2" + polar_type

    # elif pol_type_tmp == "SPPT2":
    #     n_polar_in = 2
    #     n_polar_out = 4
    #     pol_type = "SPP"
    #     pol_type_in = "SPP" + polar_type
    #     pol_type_out = "T2" + polar_type

    # elif pol_type_tmp == "SPPT4":
    #     n_polar_in = 2
    #     n_polar_out = 16
    #     pol_type = "SPP"
    #     pol_type_in = "SPP" + polar_type
    #     pol_type_out = "T4"

    # elif pol_type_tmp == "SPPIPP":
    #     n_polar_in = 2
    #     n_polar_out = 2
    #     pol_type = "SPP"
    #     pol_type_in = "SPP" + polar_type
    #     pol_type_out = "IPP"
    #     if polar_type == "pp1":
    #         pol_type_out += "pp5"
    #     elif polar_type == "pp2":
    #         pol_type_out += "pp6"
    #     elif polar_type == "pp3":
    #         pol_type_out += "pp7"

    # elif pol_type_tmp == "IPP":
    #     if polar_type == "full":
    #         n_polar_in = 4
    #         n_polar_out = 4
    #     elif polar_type in ["pp4", "pp5", "pp6", "pp7"]:
    #         n_polar_in = 2
    #         n_polar_out = 2
    #     pol_type_in = "IPP" + polar_type
    #     pol_type_out = "IPP" + polar_type

    elif pol_type_tmp == "Ixx":
        n_polar_in = 1
        n_polar_out = 1
        pol_type_in = "Ixx"
        pol_type_out = "Ixx"

    return pol_type, n_polar_in, pol_type_in, n_polar_out, pol_type_out


# %% [codecell] init_file_name
def init_file_name(pol_type, file_dir):
    """Initialisation of the binary file names"""
    file_name = []

    file_name_c2 = ["C11.bin", "C12_real.bin", "C12_imag.bin", "C22.bin"]

    file_name_c3 = [
        "C11.bin",
        "C12_real.bin",
        "C12_imag.bin",
        "C13_real.bin",
        "C13_imag.bin",
        "C22.bin",
        "C23_real.bin",
        "C23_imag.bin",
        "C33.bin",
    ]

    file_name_c4 = [
        "C11.bin",
        "C12_real.bin",
        "C12_imag.bin",
        "C13_real.bin",
        "C13_imag.bin",
        "C14_real.bin",
        "C14_imag.bin",
        "C22.bin",
        "C23_real.bin",
        "C23_imag.bin",
        "C24_real.bin",
        "C24_imag.bin",
        "C33.bin",
        "C34_real.bin",
        "C34_imag.bin",
        "C44.bin",
    ]

    file_name_t2 = ["T11.bin", "T12_real.bin", "T12_imag.bin", "T22.bin"]

    file_name_t3 = [
        "T11.bin",
        "T12_real.bin",
        "T12_imag.bin",
        "T13_real.bin",
        "T13_imag.bin",
        "T22.bin",
        "T23_real.bin",
        "T23_imag.bin",
        "T33.bin",
    ]

    file_name_t4 = [
        "T11.bin",
        "T12_real.bin",
        "T12_imag.bin",
        "T13_real.bin",
        "T13_imag.bin",
        "T14_real.bin",
        "T14_imag.bin",
        "T22.bin",
        "T23_real.bin",
        "T23_imag.bin",
        "T24_real.bin",
        "T24_imag.bin",
        "T33.bin",
        "T34_real.bin",
        "T34_imag.bin",
        "T44.bin"
    ]

    file_name_t6 = [
        "T11.bin",
        "T12_real.bin",
        "T12_imag.bin",
        "T13_real.bin",
        "T13_imag.bin",
        "T14_real.bin",
        "T14_imag.bin",
        "T15_real.bin",
        "T15_imag.bin",
        "T16_real.bin",
        "T16_imag.bin",
        "T22.bin",
        "T23_real.bin",
        "T23_imag.bin",
        "T24_real.bin",
        "T24_imag.bin",
        "T25_real.bin",
        "T25_imag.bin",
        "T26_real.bin",
        "T26_imag.bin",
        "T33.bin",
        "T34_real.bin",
        "T34_imag.bin",
        "T35_real.bin",
        "T35_imag.bin",
        "T36_real.bin",
        "T36_imag.bin",
        "T44.bin",
        "T45_real.bin",
        "T45_imag.bin",
        "T46_real.bin",
        "T46_imag.bin",
        "T55.bin",
        "T56_real.bin",
        "T56_imag.bin",
        "T66.bin"
    ]

    file_name_s2 = ["s11.bin", "s12.bin", "s21.bin", "s22.bin"]

    file_name_i = ["I11.bin", "I12.bin", "I21.bin", "I22.bin"]

    if pol_type == "C2":
        for i in range(4):
            file_name.append(os.path.join(f"{file_dir}", f"{file_name_c2[i]}"))
    elif pol_type == "C2pp1":
        for i in range(4):
            file_name.append(os.path.join(f"{file_dir}", f"{file_name_c2[i]}"))
    elif pol_type == "C2pp2":
        for i in range(4):
            file_name.append(os.path.join(f"{file_dir}", f"{file_name_c2[i]}"))
    elif pol_type == "C2pp3":
        for i in range(4):
            file_name.append(os.path.join(f"{file_dir}", f"{file_name_c2[i]}"))
    elif pol_type == "C2lhv":
        for i in range(4):
            file_name.append(os.path.join(f"{file_dir}", f"{file_name_c2[i]}"))
    elif pol_type == "C2rhv":
        for i in range(4):
            file_name.append(os.path.join(f"{file_dir}", f"{file_name_c2[i]}"))
    elif pol_type == "C2pi4":
        for i in range(4):
            file_name.append(os.path.join(f"{file_dir}", f"{file_name_c2[i]}"))
    elif pol_type == "C3":
        for i in range(9):
            file_name.append(os.path.join(f"{file_dir}", f"{file_name_c3[i]}"))
    elif pol_type == "C4":
        for i in range(16):
            file_name.append(os.path.join(f"{file_dir}", f"{file_name_c4[i]}"))
    elif pol_type == "T2":
        for i in range(4):
            file_name.append(os.path.join(f"{file_dir}", f"{file_name_t2[i]}"))
    elif pol_type == "T2pp1":
        for i in range(4):
            file_name.append(os.path.join(f"{file_dir}", f"{file_name_t2[i]}"))
    elif pol_type == "T2pp2":
        for i in range(4):
            file_name.append(os.path.join(f"{file_dir}", f"{file_name_t2[i]}"))
    elif pol_type == "T2pp3":
        for i in range(4):
            file_name.append(os.path.join(f"{file_dir}", f"{file_name_t2[i]}"))
    elif pol_type == "T2lhv":
        for i in range(4):
            file_name.append(os.path.join(f"{file_dir}", f"{file_name_t2[i]}"))
    elif pol_type == "T2rhv":
        for i in range(4):
            file_name.append(os.path.join(f"{file_dir}", f"{file_name_t2[i]}"))
    elif pol_type == "T2pi4":
        for i in range(4):
            file_name.append(os.path.join(f"{file_dir}", f"{file_name_t2[i]}"))
    elif pol_type == "T3":
        for i in range(9):
            file_name.append(os.path.join(f"{file_dir}", f"{file_name_t3[i]}"))
    elif pol_type == "T4":
        for i in range(16):
            file_name.append(os.path.join(f"{file_dir}", f"{file_name_t4[i]}"))
    elif pol_type == "T6":
        for i in range(36):
            file_name.append(os.path.join(f"{file_dir}", f"{file_name_t6[i]}"))
    elif pol_type == "S2":
        for i in range(4):
            file_name.append(os.path.join(f"{file_dir}", f"{file_name_s2[i]}"))
    elif pol_type == "SPPpp1":
        file_name.append(os.path.join(f"{file_dir}", f"{file_name_s2[0]}"))
        file_name.append(os.path.join(f"{file_dir}", f"{file_name_s2[2]}"))
    elif pol_type == "SPPpp2":
        file_name.append(os.path.join(f"{file_dir}", f"{file_name_s2[1]}"))
        file_name.append(os.path.join(f"{file_dir}", f"{file_name_s2[3]}"))
    elif pol_type == "SPPpp3":
        file_name.append(os.path.join(f"{file_dir}", f"{file_name_s2[0]}"))
        file_name.append(os.path.join(f"{file_dir}", f"{file_name_s2[3]}"))
    elif pol_type == "SPPlhv":
        file_name.append(os.path.join(f"{file_dir}", f"{file_name_s2[0]}"))
        file_name.append(os.path.join(f"{file_dir}", f"{file_name_s2[2]}"))
    elif pol_type == "SPPrhv":
        file_name.append(os.path.join(f"{file_dir}", f"{file_name_s2[0]}"))
        file_name.append(os.path.join(f"{file_dir}", f"{file_name_s2[2]}"))
    elif pol_type == "SPPpi4":
        file_name.append(os.path.join(f"{file_dir}", f"{file_name_s2[0]}"))
        file_name.append(os.path.join(f"{file_dir}", f"{file_name_s2[2]}"))
    elif pol_type == "IPPfull":
        for i in range(4):
            file_name.append(os.path.join(f"{file_dir}", f"{file_name_i[i]}"))
    elif pol_type == "IPPpp4":
        file_name.append(os.path.join(f"{file_dir}", f"{file_name_i[0]}"))
        file_name.append(os.path.join(f"{file_dir}", f"{file_name_i[1]}"))
        file_name.append(os.path.join(f"{file_dir}", f"{file_name_i[3]}"))
    elif pol_type == "IPPpp5":
        file_name.append(os.path.join(f"{file_dir}", f"{file_name_i[0]}"))
        file_name.append(os.path.join(f"{file_dir}", f"{file_name_i[2]}"))
    elif pol_type == "IPPpp6":
        file_name.append(os.path.join(f"{file_dir}", f"{file_name_i[1]}"))
        file_name.append(os.path.join(f"{file_dir}", f"{file_name_i[3]}"))
    elif pol_type == "IPPpp7":
        file_name.append(os.path.join(f"{file_dir}", f"{file_name_i[0]}"))
        file_name.append(os.path.join(f"{file_dir}", f"{file_name_i[3]}"))

    return file_name


# %% [codecell] read_config
def read_config(file_dir):
    """Read a configuration file"""
    nlig = 0
    ccol = 0
    polar_case = ""
    polar_type = ""

    # if os.path.exists(os.path.join(file_dir, "config.txt")):
    with open(os.path.join(file_dir, "config.txt"), "r") as file:
        for line in file:
            if "Nrow" in line:
                nlig = int(file.readline().strip())
            elif "Ncol" in line:
                ccol = int(file.readline().strip())
            elif "PolarCase" in line:
                polar_case = file.readline().strip()
            elif "PolarType" in line:
                polar_type = file.readline().strip()
    return nlig, ccol, polar_case, polar_type


class Pix:
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
        self.next: None


def Create_Pix(P, x, y):
    '''
    Structure: Create_Pix
    Authors  : Laurent FERRO-FAMIL
    Creation : 07/2003
    Update  :
    *--------------------------------------------------------------------
    Description :
    ********************************************************************
    '''
    if P is None:
        P = Pix()
        P.x = x
        P.y = y
        P.next = None
    else:
        raise 'Error Create Pix'
    return P


def Remove_Pix(P_top, P):
    '''
    Structure: Remove_Pix
    Authors  : Laurent FERRO-FAMIL
    Creation : 07/2003
    Update  :
    *--------------------------------------------------------------------
    Description :
    ********************************************************************
    '''
    P_current: Pix

    if P is None:
        raise 'Error Create Pix'
    if P == P_top:
        P_current = P_top
        P = P.next
    else:
        if P.next is None:
            P_current = P_top
            while P_current.next != P:
                P_current = P_current.next
            P = P_current
            P_current = P_current.next
        else:
            P_current = P_top
            while P_current.next != P:
                P_current = P_current.next
            P = P_current
            P_current = P_current.next
            P.next = P_current.next
    P_current = None
    return P


@numba.njit()
def my_fseek(in_file, fseek_sign, fseek_arg1, fseek_arg2):
    """
    Description : Function fseek with a pointer size higher than LONG_MAX
    """
    # in_file.seek((fseek_sign) * (fseek_arg1) * (fseek_arg2), 1)
    pos = in_file.tell()
    pos = pos + (fseek_sign) * (fseek_arg1) * (fseek_arg2)
    in_file.seek(pos, 0)


@numba.njit()
def printf_line(lig, NNlig):
    if NNlig > 20:
        if lig % (int)(NNlig / 20) == 0:
            # sf = 100. * lig / (NNlig - 1)
            with numba.objmode():
                print("{:.2f}%\r".format(100. * lig / (NNlig - 1)), end="", flush=True)
    else:
        if NNlig > 1:
            # sf = 100. * lig / (NNlig - 1)
            with numba.objmode():
                print("{:.2f}%\r".format(100. * (lig + 1) / NNlig), end="", flush=True)
