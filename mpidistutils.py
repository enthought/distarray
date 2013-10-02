# Author:    Lisandro Dalcin
# Contact:   dalcinl@gmail.com
# Copyright: This module has been placed in the public domain.
# Id:        $Id: mpidistutils.py 152 2007-11-28 20:44:03Z dalcinl $

"""
Support for building mpi4py with distutils.
"""

# --------------------------------------------------------

# Environmental variables to look for configuration
MPICC_ENV  = ['MPI4PY_CC',  'MPICC']
MPICXX_ENV = ['MPI4PY_CXX', 'MPICXX']
MPICFG_ENV = ['MPI4PY_CFG', 'MPI_CONFIG', 'MPI_CFG']

# Default values to use for configuration
MPICC  = ['mpicc']
MPICXX = ['mpicxx', 'mpiCC', 'mpic++']
MPICFG = ('mpi', 'mpi.cfg')

# --------------------------------------------------------

import sys, os
from distutils import sysconfig
from distutils.spawn import find_executable
from distutils import log


def customize_compiler(compiler, environ=None):
    """
    Do any platform-specific customization of a CCompiler instance.

    Mainly needed on Unix, so we can plug in the information that
    varies across Unices and is stored in Python's Makefile.
    """
    from distutils.sysconfig import get_config_vars
    if environ is None:
        environ = os.environ

    if compiler.compiler_type == 'unix':
        (cc, cxx, opt,
         basecflags, cflags, ccshared,
         ldshared, so_ext) = \
            get_config_vars('CC', 'CXX',
                            'BASECFLAGS', 'OPT', 'CFLAGS',
                            'CCSHARED', 'LDSHARED', 'SO')

        if 'CC' in environ:
            cc = environ['CC']
        if 'CXX' in environ:
            cxx = environ['CXX']
        if 'LDSHARED' in environ:
            ldshared = environ['LDSHARED']
        if 'CPP' in environ:
            cpp = environ['CPP']
        else:
            cpp = cc + " -E"           # not always
        if 'LDFLAGS' in environ:
            ldshared = ldshared + ' ' + environ['LDFLAGS']
        if 'CFLAGS' in environ:
            cflags = basecflags + ' ' + opt + ' ' + environ['CFLAGS']
            ldshared = ldshared + ' ' + environ['CFLAGS']
        if 'CPPFLAGS' in environ:
            cpp = cpp + ' ' + environ['CPPFLAGS']
            cflags = cflags + ' ' + environ['CPPFLAGS']
            ldshared = ldshared + ' ' + environ['CPPFLAGS']

        cc_cmd = cc + ' ' + cflags
        compiler.set_executables(
            preprocessor=cpp,
            compiler=cc_cmd,
            compiler_so=cc_cmd + ' ' + ccshared,
            compiler_cxx=cxx,
            linker_so=ldshared,
            linker_exe=cc)

        compiler.shared_lib_extension = so_ext

def customize_mpi_environ(mpicc, mpicxx=None, environ=None):
    """
    Replace normal compilers with MPI compilers
    """
    if environ is None:
        environ = dict(os.environ)
    if mpicc: # C compiler
        environ['CC'] = mpicc
    if mpicxx: # C++ compiler
        environ['CXX'] = mpicxx
    mpild = mpicc or mpicxx
    if mpild: # linker for shared
        ldshared = sysconfig.get_config_var('LDSHARED')
        if not ldshared:
            environ['LDSHARED'] = mpild
        else:
            if sys.platform.startswith('aix') \
                   and 'ld_so_aix' in ldshared:
                ldshared = ldshared.split(' ', 2)
            else:
                ldshared = ldshared.split(' ', 1)
            if len(ldshared) == 1: # just linker, no flags
                environ['LDSHARED'] = mpild
            elif len(ldshared) == 2: # linker and flags
                environ['LDSHARED'] = mpild + ' ' + ldshared[1]
            else: # assume using special linker script
                environ['LDSHARED'] = ldshared[0] + ' '  + \
                                      mpild       + ' '  + \
                                      ldshared[2]
    return environ

def _find_mpi_compiler(envvars, executables, path=None):
    """
    Find MPI compilers in environment and path.
    """
    # search in environment
    if envvars:
        if isinstance(envvars, str):
            envvars = (envvars,)
        for var in envvars:
            cmd = os.environ.get(var)
            if cmd is not None: return cmd
    # search in path
    if executables:
        if isinstance(executables, str):
            executables = (executables,)
        for exe in executables:
            try:
                cmd, args = exe.split(' ', 1)
            except ValueError:
                cmd, args = exe, None
            cmd = find_executable(cmd, path)
            if cmd is not None:
                if args is not None:
                    cmd = cmd + ' ' + args
                return cmd
    # nothing found
    return None


# --------------------------------------------------------

from six.moves import configparser

def _config_parser(section, filenames, raw=False, vars=None):
    """
    Returns a dictionary of options obtained by parsing configuration
    files.
    """
    parser = configparser.ConfigParser()
    try:
        parser.read(filenames.split(','))
    except configparser.Error:
        log.error("error: parsing configuration file/s '%s'", filenames)
        return None
    if not parser.has_section(section):
        log.error("error: section '%s' not found "
                  "in configuration file/s '%s'", section, filenames)
        return None
    config_info = {}
    for k, v in parser.items(section, raw, vars):
        if k in ('define_macros',
                 'undef_macros',):
            config_info[k] = [m.strip() for m in v.split()]
        elif k in ('include_dirs',
                   'library_dirs',
                   'runtime_library_dirs',):
            pathsep = os.path.pathsep
            config_info[k] = [p.strip() for p in v.split(pathsep)]
        elif k == 'libraries':
            config_info[k] = [l.strip() for l in v.split()]
        elif k in ('extra_compile_args',
                   'extra_link_args',
                   'extra_objects',):
            config_info[k] = [e.strip() for e in v.split()]
        else:
            config_info[k] = v.strip()
        #config_info[k] = v.replace('\\',' ').split()
    if 'define_macros' in config_info:
        macros = []
        for m in config_info['define_macros'] :
            try: # "-DFOO=blah"
                idx = m.index("=")
                macro = (m[:idx], m[idx+1:] or None)
            except ValueError: # bare "-DFOO"
                macro = (m, None)
            macros.append(macro)
        config_info['define_macros'] = macros
    return config_info


def _find_mpi_config(section, filenames,
                     envvars=None, defaults=None):
    if not section and not filenames and envvars:
        # look in environment
        if isinstance(envvars, str):
            envvars = (envvars,)
        for var in envvars:
            if var in os.environ:
                section, filenames = os.environ[var], None
                break
    if section and ',' in section:
        section, filenames = section.split(',', 1)
    if defaults:
        if not section:
            section = defaults[0]
        if not filenames:
            fname = defaults[1]
            if os.path.exists(fname):
                filenames = fname
    # parse configuraration
    if section and filenames:
        config_info = _config_parser(section, filenames)
        return section, filenames, config_info
    else:
        return section, filenames, None


def _configure(extension, confdict):
    if confdict is None: return
    for key, value in confdict.items():
        if hasattr(extension, key):
            item = getattr(extension, key)
            if type(item) is list:
                if type(value) is list:
                    for v in value:
                        if v not in item:
                            item.append(v)
                else:
                    if value not in item:
                        item.append(value)
            else:
                setattr(extension, key, value)

# --------------------------------------------------------

from distutils.command import config as cmd_config
from distutils.command import build as cmd_build
from distutils.command import build_ext as cmd_build_ext
from distutils.command import sdist as cmd_sdist

from distutils.errors import DistutilsPlatformError
from distutils.errors import DistutilsOptionError


ConfigTests = [
    # ----------------------------------------------------------------

    ('MPI_VERSION',
     'int version; version = MPI_VERSION;'),
    ('MPI_GET_VERSION',
     'MPI_Get_version((int*)0,(int*)0);'),

    # ----------------------------------------------------------------

    ('MPI_INIT_THREAD',
     "MPI_Init_thread((int*)0,(char***)0,MPI_THREAD_SINGLE,(int*)0);"),
    ('MPI_QUERY_THREAD',
     "MPI_Query_thread((int*)0);"),
    ('MPI_IS_THREAD_MAIN',
     "MPI_Is_thread_main((int*)0);"),

    # ----------------------------------------------------------------

    ('MPI_WCHAR',
     'MPI_Datatype datatype; datatype = MPI_WCHAR;'),
    ('MPI_SIGNED_CHAR',
     'MPI_Datatype datatype; datatype = MPI_SIGNED_CHAR;'),
    ('MPI_LONG_LONG',
     'MPI_Datatype datatype; datatype = MPI_LONG_LONG;'),
    ('MPI_LONG_LONG_INT',
     'MPI_Datatype datatype; datatype = MPI_LONG_LONG_INT;'),
    ('MPI_UNSIGNED_LONG_LONG',
     'MPI_Datatype datatype; datatype = MPI_UNSIGNED_LONG_LONG;'),
    ('MPI_TYPE_GET_EXTENT',
     'MPI_Type_get_extent(MPI_DATATYPE_NULL,(MPI_Aint*)0,(MPI_Aint*)0);'),
    ('MPI_TYPE_DUP',
     'MPI_Type_dup(MPI_DATATYPE_NULL,(MPI_Datatype*)0);'),
    ('MPI_TYPE_CREATE_INDEXED_BLOCK',
     'MPI_Type_create_indexed_block(0,0,(int*)0,MPI_DATATYPE_NULL,(MPI_Datatype*)0);'),
    ('MPI_TYPE_CREATE_HVECTOR',
     'MPI_Type_create_hvector(0,0,(MPI_Aint)0,MPI_DATATYPE_NULL,(MPI_Datatype*)0);'),
    ('MPI_TYPE_CREATE_HINDEXED',
     'MPI_Type_create_hindexed(0,(int*)0,(MPI_Aint*)0,MPI_DATATYPE_NULL,(MPI_Datatype*)0);'),
    ('MPI_TYPE_CREATE_STRUCT',
     'MPI_Type_create_struct(0,(int*)0,(MPI_Aint*)0,(MPI_Datatype*)0,(MPI_Datatype*)0);'),
    ('MPI_TYPE_CREATE_SUBARRAY',
     'MPI_Type_create_subarray(0,(int*)0,(int*)0,(int*)0,0,MPI_DATATYPE_NULL,(MPI_Datatype*)0);'),
    ('MPI_TYPE_CREATE_DARRAY',
     'MPI_Type_create_darray(0,0,0,(int*)0,(int*)0,(int*)0,(int*)0,0,MPI_DATATYPE_NULL,(MPI_Datatype*)0);'),
    ('MPI_TYPE_CREATE_RESIZED',
     'MPI_Type_create_resized(MPI_DATATYPE_NULL,0,0,(MPI_Datatype*)0);'),
    ('MPI_TYPE_GET_TRUE_EXTENT',
     'MPI_Type_get_true_extent(MPI_DATATYPE_NULL,(MPI_Aint*)0,(MPI_Aint*)0);'),
    ('MPI_GET_ADDRESS',
     'MPI_Get_address((void*)0,(MPI_Aint*)0);'),
    ('MPI_PACK_EXTERNAL',
     'MPI_Pack_external((char*)0,(void*)0,0,MPI_DATATYPE_NULL,(void*)0,0,(MPI_Aint*)0);'),
    ('MPI_UNPACK_EXTERNAL',
     'MPI_Unpack_external((char*)0,(void*)0,0,(MPI_Aint*)0,(void*)0,0,MPI_DATATYPE_NULL);'),
    ('MPI_PACK_EXTERNAL_SIZE',
     'MPI_Pack_external_size((char*)0,0,MPI_DATATYPE_NULL,(MPI_Aint*)0);'),

    # ----------------------------------------------------------------

    ('MPI_STATUS_IGNORE' ,
     'MPI_Status *status; status = MPI_STATUS_IGNORE;'),
    ('MPI_STATUSES_IGNORE' ,
     'MPI_Status *status; status = MPI_STATUSES_IGNORE;'),
    ('MPI_STATUS_SET_ELEMENTS' ,
     'MPI_Status s; MPI_Status_set_elements(&s,MPI_BYTE,0);'),
    ('MPI_STATUS_SET_CANCELLED' ,
     'MPI_Status s; MPI_Status_set_cancelled(&s,0);'),
    ('MPI_REQUEST_GET_STATUS' ,
     'MPI_Request_get_status(MPI_REQUEST_NULL,(int*)0,(MPI_Status*)0);'),
    ('MPI_GREQUEST_START' ,
     'MPI_Request r; MPI_Grequest_start(0,0,0,0,&r);'),
    ('MPI_GREQUEST_COMPLETE' ,
     'MPI_Grequest_complete(MPI_REQUEST_NULL);'),

    # ----------------------------------------------------------------

    ('MPI_IN_PLACE',
     'MPI_Scan(MPI_IN_PLACE,(void*)0,0,MPI_DATATYPE_NULL,MPI_OP_NULL,MPI_COMM_NULL);'),
    ('MPI_ROOT',
     'MPI_Bcast((void*)0,0,MPI_DATATYPE_NULL,MPI_ROOT,MPI_COMM_NULL);'),
    ('MPI_EXSCAN',
     'MPI_Exscan((void*)0,(void*)0,0,MPI_DATATYPE_NULL,MPI_OP_NULL,MPI_COMM_NULL);'),

    # ----------------------------------------------------------------

    ('MPI_COMM_GET_ERRHANDLER' ,
     'MPI_Comm_get_errhandler(MPI_COMM_NULL,(MPI_Errhandler*)0);'),
    ('MPI_COMM_SET_ERRHANDLER' ,
     'MPI_Comm_set_errhandler(MPI_COMM_NULL,MPI_ERRHANDLER_NULL);'),
    ('MPI_COMM_CALL_ERRHANDLER',
     'MPI_Comm_call_errhandler(MPI_COMM_NULL,0);'),

    # ----------------------------------------------------------------

    ('MPI_INFO_NULL',
     'MPI_Info info; info = MPI_INFO_NULL;'),

    ('MPI_INFO_CREATE',
     'MPI_Info_create((MPI_Info*)0);'),
    ('MPI_INFO_FREE',
     'MPI_Info_free((MPI_Info*)0);'),

    ('MPI_ERR_INFO',       'int ierr; ierr = MPI_ERR_INFO;'),
    ('MPI_ERR_INFO_KEY',   'int ierr; ierr = MPI_ERR_INFO_KEY;'),
    ('MPI_ERR_INFO_VALUE', 'int ierr; ierr = MPI_ERR_INFO_VALUE;'),
    ('MPI_ERR_INFO_NOKEY', 'int ierr; ierr = MPI_ERR_INFO_NOKEY;'),

    ('MPI_MAX_INFO_KEY',   'int imxl; imxl = MPI_MAX_INFO_KEY;'),
    ('MPI_MAX_INFO_VAL',   'int imxl; imxl = MPI_MAX_INFO_KEY;'),

    # ----------------------------------------------------------------

    ('MPI_WIN_NULL',
     'MPI_Win win; win = MPI_WIN_NULL;'),

    ('MPI_WIN_CREATE',
     'MPI_Win_create((void*)0,(MPI_Aint)0,0,MPI_INFO_NULL,MPI_COMM_NULL,(MPI_Win*)0);'),
    ('MPI_WIN_FREE',
     'MPI_Win_free((MPI_Win*)0);'),

    ('MPI_WIN_TEST',
     'MPI_Win_test(MPI_WIN_NULL,(int*)0);'),
    ('MPI_WIN_LOCK',
     'MPI_Win_lock(0,0,0,MPI_WIN_NULL);'),
    ('MPI_WIN_UNLOCK',
     'MPI_Win_unlock(0,MPI_WIN_NULL);'),

    ('MPI_LOCK_EXCLUSIVE'  , 'int lock; lock = MPI_LOCK_EXCLUSIVE;'  ),
    ('MPI_LOCK_SHARED'     , 'int lock; lock = MPI_LOCK_SHARED;'     ),

    ('MPI_ERR_WIN'         , 'int ierr; ierr = MPI_ERR_WIN;'         ),
    ('MPI_ERR_BASE'        , 'int ierr; ierr = MPI_ERR_BASE;'        ),
    ('MPI_ERR_SIZE'        , 'int ierr; ierr = MPI_ERR_SIZE;'        ),
    ('MPI_ERR_DISP'        , 'int ierr; ierr = MPI_ERR_DISP;'        ),
    ('MPI_ERR_LOCKTYPE'    , 'int ierr; ierr = MPI_ERR_LOCKTYPE;'    ),
    ('MPI_ERR_ASSERT'      , 'int ierr; ierr = MPI_ERR_ASSERT;'      ),
    ('MPI_ERR_RMA_CONFLICT', 'int ierr; ierr = MPI_ERR_RMA_CONFLICT;'),
    ('MPI_ERR_RMA_SYNC'    , 'int ierr; ierr = MPI_ERR_RMA_SYNC;'    ),

    ('MPI_REPLACE', 'MPI_Op op; op = MPI_REPLACE;'),

    # ----------------------------------------------------------------

    ('MPI_FILE_NULL',
     'MPI_File file; file = MPI_FILE_NULL;'),

    ('MPI_FILE_OPEN',
     'MPI_File_open(MPI_COMM_NULL,(char*)0,0,MPI_INFO_NULL,(MPI_File*)0);'),
    ('MPI_FILE_CLOSE',
     'MPI_File_close((MPI_File*)0);'),

    ('MPI_MODE_RDONLY',               'int mode; mode = MPI_MODE_RDONLY;'),
    ('MPI_MODE_RDWR',                 'int mode; mode = MPI_MODE_RDWR;'),
    ('MPI_MODE_WRONLY',               'int mode; mode = MPI_MODE_WRONLY;'),
    ('MPI_MODE_CREATE',               'int mode; mode = MPI_MODE_CREATE;'),
    ('MPI_MODE_EXCL',                 'int mode; mode = MPI_MODE_EXCL;'),
    ('MPI_MODE_DELETE_ON_CLOSE',      'int mode; mode = MPI_MODE_DELETE_ON_CLOSE;'),
    ('MPI_MODE_UNIQUE_OPEN',          'int mode; mode = MPI_MODE_UNIQUE_OPEN;'),
    ('MPI_MODE_SEQUENTIAL',           'int mode; mode = MPI_MODE_SEQUENTIAL;'),
    ('MPI_MODE_APPEND',               'int mode; mode = MPI_MODE_APPEND;'),

    ('MPI_SEEK_SET',                  'int seek; seek = MPI_SEEK_SET;'),
    ('MPI_SEEK_CUR',                  'int seek; seek = MPI_SEEK_CUR;'),
    ('MPI_SEEK_END',                  'int seek; seek = MPI_SEEK_END;'),

    ('MPI_ERR_FILE',                  'int ierr; ierr = MPI_ERR_FILE;'),
    ('MPI_ERR_NOT_SAME',              'int ierr; ierr = MPI_ERR_NOT_SAME;'),
    ('MPI_ERR_AMODE',                 'int ierr; ierr = MPI_ERR_AMODE;'),
    ('MPI_ERR_UNSUPPORTED_DATAREP',   'int ierr; ierr = MPI_ERR_UNSUPPORTED_DATAREP;'),
    ('MPI_ERR_UNSUPPORTED_OPERATION', 'int ierr; ierr = MPI_ERR_UNSUPPORTED_OPERATION;'),
    ('MPI_ERR_NO_SUCH_FILE',          'int ierr; ierr = MPI_ERR_NO_SUCH_FILE;'),
    ('MPI_ERR_FILE_EXISTS',           'int ierr; ierr = MPI_ERR_FILE_EXISTS;'),
    ('MPI_ERR_BAD_FILE',              'int ierr; ierr = MPI_ERR_BAD_FILE;'),
    ('MPI_ERR_ACCESS',                'int ierr; ierr = MPI_ERR_ACCESS;'),
    ('MPI_ERR_NO_SPACE',              'int ierr; ierr = MPI_ERR_NO_SPACE;'),
    ('MPI_ERR_QUOTA',                 'int ierr; ierr = MPI_ERR_QUOTA;'),
    ('MPI_ERR_READ_ONLY',             'int ierr; ierr = MPI_ERR_READ_ONLY;'),
    ('MPI_ERR_FILE_IN_USE',           'int ierr; ierr = MPI_ERR_FILE_IN_USE;'),
    ('MPI_ERR_DUP_DATAREP',           'int ierr; ierr = MPI_ERR_DUP_DATAREP;'),
    ('MPI_ERR_CONVERSION',            'int ierr; ierr = MPI_ERR_CONVERSION;'),
    ('MPI_ERR_IO',                    'int ierr; ierr = MPI_ERR_IO;'),

    ('MPI_DISPLACEMENT_CURRENT', 'MPI_Offset val; val = MPI_DISPLACEMENT_CURRENT;'),
    ('MPI_MAX_DATAREP_STRING',   'int val;        val = MPI_MAX_DATAREP_STRING;'),

    # ----------------------------------------------------------------

    ('MPI_ALLOC_MEM',
     'MPI_Alloc_mem((MPI_Aint)0,MPI_INFO_NULL,(void*)0);'),
    ('MPI_FREE_MEM',
     'MPI_Free_mem((void*)0);'),

    ('MPI_ERR_NO_MEM', 'int ierr; ierr = MPI_ERR_NO_MEM;'),

    # ----------------------------------------------------------------

    ('MPI_OPEN_PORT',
     'MPI_Open_port(MPI_INFO_NULL,(char*)0);'),
    ('MPI_CLOSE_PORT',
     'MPI_Close_port((char*)0);'),
    ('MPI_PUBLISH_NAME',
     'MPI_Publish_name((char*)0, MPI_INFO_NULL,(char*)0);'),
    ('MPI_UNPUBLISH_NAME',
     'MPI_Unpublish_name((char*)0, MPI_INFO_NULL,(char*)0);'),
    ('MPI_LOOKUP_NAME',
     'MPI_Lookup_name((char*)0, MPI_INFO_NULL,(char*)0);'),

    ('MPI_COMM_ACCEPT',
     'MPI_Comm_accept((char*)0,MPI_INFO_NULL,0,MPI_COMM_NULL,(MPI_Comm*)0);'),
    ('MPI_COMM_CONNECT',
     'MPI_Comm_connect((char*)0,MPI_INFO_NULL,0,MPI_COMM_NULL,(MPI_Comm*)0);'),
    ('MPI_COMM_DISCONNECT',
     'MPI_Comm comm = MPI_COMM_NULL; MPI_Comm_disconnect(&comm);'),
    ('MPI_COMM_SPAWN',
     'MPI_Comm_spawn((char*)0,(char**)0,0,' \
     'MPI_INFO_NULL,0,MPI_COMM_NULL,(MPI_Comm*)0,(int*)0);'),
    ('MPI_COMM_GET_PARENT',
     'MPI_Comm_get_parent((MPI_Comm*)0);'),
    ('MPI_COMM_JOIN',
     'MPI_Comm_join(0,(MPI_Comm*)0);'),

    ('MPI_ERRCODES_IGNORE', 'int *eci; eci  = MPI_ERRCODES_IGNORE;'),
    ('MPI_APPNUM',          'int kval; kval = MPI_APPNUM;'),
    ('MPI_UNIVERSE_SIZE',   'int kval; kval = MPI_UNIVERSE_SIZE;'),

    ('MPI_ERR_NAME',    'int ierr; ierr = MPI_ERR_NAME;'),
    ('MPI_ERR_PORT',    'int ierr; ierr = MPI_ERR_PORT;'),
    ('MPI_ERR_SERVICE', 'int ierr; ierr = MPI_ERR_SERVICE;'),
    ('MPI_ERR_SPAWN',   'int ierr; ierr = MPI_ERR_SPAWN;'),

    # ----------------------------------------------------------------

    ('MPI_TYPE_GET_NAME',
     'MPI_Type_get_name(MPI_DATATYPE_NULL,(char*)0,(int*)0);'),
    ('MPI_TYPE_SET_NAME',
     'MPI_Type_set_name(MPI_DATATYPE_NULL,(char*)0);'),
    ('MPI_COMM_GET_NAME',
     'MPI_Comm_get_name(MPI_COMM_NULL,(char*)0,(int*)0);'),
    ('MPI_COMM_SET_NAME',
     'MPI_Comm_set_name(MPI_COMM_NULL,(char*)0);'),
    ('MPI_WIN_GET_NAME',
     'MPI_Win_get_name(MPI_WIN_NULL,(char*)0,(int*)0);'),
    ('MPI_WIN_SET_NAME',
     'MPI_Win_set_name(MPI_WIN_NULL,(char*)0);'),
    ('MPI_MAX_OBJECT_NAME', 'int val; val = MPI_MAX_OBJECT_NAME;'),

    # ----------------------------------------------------------------

    ('MPI_ERR_KEYVAL', 'int ierr; ierr = MPI_ERR_KEYVAL;'),

    # ----------------------------------------------------------------
    ]

_ConfigTest = """\
int main(int argc, char **argv) {
  MPI_Init(&argc,&argv);
  MPI_Finalize();
  return 0;
}
"""


cmd_mpi_opts = [

    ('mpicc=',   None,
     "MPI C compiler command, "
     "overrides environmental variables 'MPICC' "
     "(defaults to 'mpicc' if available)"),

    ('mpicxx=',  None,
     "MPI C++ compiler command, "
     "overrides environmental variables 'MPICXX' "
     "(defaults to 'mpicxx', 'mpiCC', or 'mpic++' if any is available)"),

    ('mpi-cfg=', None,
     "specify a configuration file to look for MPI includes/libraries "
     "(defaults to 'mpi.cfg')"),

    ('mpi=',     None,
     "specify a configuration section, "
     "and an optional comma-separated list of configuration files "
     "(e.g. --mpi=section,file1,file2,file3),"
     "to look for MPI includes/libraries, "
     "overrides environmental variables 'MPICFG' "
     "(defaults to section 'mpi' in configuration file 'mpi.cfg')"),

    ('try-mpi-2', None,
     "test for availability of MPI 2 features"),

    ('thread-level=', None,
     "initialize MPI with support for threads"),

    ('enable-fast-errcheck', None,
     "faster error checking (by forcing  MPI_ERRORS_RETURN)"),

    ('enable-leak-warning', None,
     "isuue a warning if any MPI handle is (possibly) leaked"),

    ]

def _cmd_opts_names(cmd_opts):
    optlist = []
    for (option, _, _) in cmd_opts:
        if option[-1] == "=":
            option = option[:-1]
        option = option.replace('-','_')
        optlist.append(option)
    return optlist

def _thread_level(thread_level):
    if not thread_level:
        return None
    thread_level = thread_level.lower()
    if thread_level == 'none':
        return None
    valid_levels = ['single', 'funneled', 'serialized', 'multiple']
    if thread_level not in valid_levels:
        raise DistutilsOptionError(
            ("thread-level must be one of " + ','.join(valid_levels))
            )
    return thread_level

def _enable_fast_errcheck(flag):
    if flag:
        return 1
    else:
        return 0


class config(cmd_config.config):

    user_options = cmd_config.config.user_options + cmd_mpi_opts

    def initialize_options(self):
        cmd_config.config.initialize_options(self)
        self.noisy = 0
        mpiopts = _cmd_opts_names(cmd_mpi_opts)
        for op in mpiopts:
            setattr(self, op, None)

    def finalize_options(self):
        cmd_config.config.finalize_options(self)
        if not self.noisy:
            self.dump_source = 0
        self.thread_level = _thread_level(self.thread_level)

    def find_mpi_compiler(self, envvars, executables, path=None):
        return _find_mpi_compiler(envvars, executables, path)

    def run(self):
        # test MPI C compiler
        mpicc = self.mpicc
        if mpicc is None:
            mpicc = self.find_mpi_compiler(MPICC_ENV, MPICC)
        if mpicc:
            log.info("MPI C compiler:    %s", mpicc  or 'not found')
            self.compiler = None
            self._check_compiler()
            environ = customize_mpi_environ(mpicc, None)
            customize_compiler(self.compiler, environ)
            self.try_link(_ConfigTest, headers=['mpi.h'], lang='c')
        # test MPI C++ compiler
        mpicxx = self.mpicxx
        if mpicxx is None:
            mpicxx = self.find_mpi_compiler(MPICXX_ENV, MPICXX)
        if mpicxx:
            log.info("MPI C++ compiler:  %s", mpicxx or 'not found')
            self.compiler = None
            self._check_compiler()
            environ = customize_mpi_environ(None, mpicxx)
            customize_compiler(self.compiler, environ)
            if self.compiler.compiler_type in ('unix', 'cygwin', 'mingw32'):
                self.compiler.compiler_so[0] = mpicxx
                self.compiler.linker_exe[0] = mpicxx
            self.try_link(_ConfigTest, headers=['mpi.h'], lang='c++')
        # test configuration in specified section and file
        if self.mpi or self.mpi_cfg:
            sct, fn, cfg = _find_mpi_config(self.mpi, self.mpi_cfg,
                                            MPICFG_ENV, MPICFG)
            log.info("MPI configuration: "
                     "section '%s' from file/s '%s'", sct, fn)
            _configure(self, cfg)
            self.compiler = None
            self._check_compiler()
            self.try_link(_ConfigTest, headers=['mpi.h'], lang='c')

    def get_config_macros(self, test_list, compiler, config_info):
        self.compiler = compiler
        _configure(self, config_info)
        # run configure tests
        macros = []
        for name, code in test_list:
            result = self.try_config_test(code)
            macros.append(('HAVE_%s' % name, int(result)))
        return macros

    def try_config_test(self, lines):
        if type(lines) is not str:
            lines = '\n  '.join(lines)
        body = "int main(void) {\n  %s\n  return 0;\n}\n" % lines
        return self.try_link(body, headers=['mpi.h'],
                             include_dirs=self.include_dirs,
                             libraries=self.libraries,
                             library_dirs=self.library_dirs)


class build(cmd_build.build):

    user_options = cmd_build.build.user_options + cmd_mpi_opts

    def initialize_options(self):
        cmd_build.build.initialize_options(self)
        mpiopts = _cmd_opts_names(cmd_mpi_opts)
        for op in mpiopts:
            setattr(self, op, None)

    def finalize_options(self):
        cmd_build.build.finalize_options(self)
        config_cmd = self.get_finalized_command('config')
        if isinstance(config_cmd,  config):
            mpiopts = _cmd_opts_names(cmd_mpi_opts)
            optlist = tuple(zip(mpiopts, mpiopts))
            self.set_undefined_options('config', *optlist)
        self.thread_level = _thread_level(self.thread_level)


class build_ext(cmd_build_ext.build_ext):

    user_options = cmd_build_ext.build_ext.user_options + cmd_mpi_opts

    def initialize_options(self):
        cmd_build_ext.build_ext.initialize_options(self)
        mpiopts = _cmd_opts_names(cmd_mpi_opts)
        for op in mpiopts:
            setattr(self, op, None)

    def finalize_options(self):
        cmd_build_ext.build_ext.finalize_options(self)
        import sys, os
        if (sys.platform.startswith('linux') or \
            sys.platform.startswith('gnu')) and \
            sysconfig.get_config_var('Py_ENABLE_SHARED'):
            try:
                py_version = sysconfig.get_python_version()
                bad_pylib_dir = os.path.join(sys.prefix, "lib",
                                             "python" + py_version,
                                             "config")
                self.library_dirs.remove(bad_pylib_dir)
            except ValueError:
                pass
            pylib_dir = sysconfig.get_config_var("LIBDIR")
            if pylib_dir not in self.library_dirs:
                self.library_dirs.append(pylib_dir)
        build_cmd = self.get_finalized_command('build')
        if isinstance(build_cmd,  build):
            mpiopts = _cmd_opts_names(cmd_mpi_opts)
            optlist = tuple(zip(mpiopts, mpiopts))
            self.set_undefined_options('build', *optlist)
        self.thread_level = _thread_level(self.thread_level)

    def build_extensions(self):
        # First, sanity-check the 'extensions' list
        self.check_extensions_list(self.extensions)
        # parse configuration file and  configure compiler
        config_info = self.configure_extensions()
        mpicc = mpicxx = None
        if config_info:
            mpicc = config_info.get('mpicc')
            mpicxx = config_info.get('mpicxx')
        compiler = self.configure_compiler(mpicc=mpicc, mpicxx=mpicxx)
        # extra configuration, MPI 2 features
        if self.try_mpi_2:
            log.info('testing for MPI 2 features')
            config = self.get_finalized_command('config')
            get_macros = config.get_config_macros
            macros = get_macros(ConfigTests, compiler, config_info)
            for (name, value) in macros:
                self.compiler.define_macro(name, value)
                log.info('#define %s %d' % (name, value))
        # extra configuration, MPI thread level support
        if self.thread_level:
            levels = ['single', 'funneled', 'serialized', 'multiple']
            name = 'MPI4PY_THREAD_LEVEL'
            value = levels.index(self.thread_level)
            self.compiler.define_macro(name, value)
            log.info('#define %s %d' % (name, value))
        # extra configuration, leak warnings
        if self.enable_leak_warning:
            name, value = 'MPI4PY_ENABLE_LEAK_WARNING', 1
            self.compiler.define_macro(name, value)
            log.info('#define %s %d' % (name, value))
        # extra configuration, fast error checking
        if self.enable_fast_errcheck:
            name, value = 'MPI4PY_ENABLE_FAST_ERRCHECK', 1
            self.compiler.define_macro(name, value)
            log.info('#define %s %d' % (name, value))

        # build extensions
        for ext in self.extensions:
            self.build_extension(ext)

    def configure_compiler(self, compiler=None, mpicc=None, mpicxx=None):
        mpicc, mpicxx = self.mpicc or mpicc, self.mpicxx or mpicxx
        if mpicc is None:
            mpicc = self.find_mpi_compiler(MPICC_ENV, MPICC)
        if mpicxx is None:
            mpicxx = self.find_mpi_compiler(MPICXX_ENV, MPICXX)
        if compiler is None:
            compiler = self.compiler
        log.info("MPI C compiler:    %s", mpicc  or 'not found')
        log.info("MPI C++ compiler:  %s", mpicxx or 'not found')
        environ = customize_mpi_environ(mpicc, mpicxx)
        customize_compiler(compiler, environ)
        return compiler

    def find_mpi_compiler(self, envvars, executables, path=None):
        return _find_mpi_compiler(envvars, executables, path)

    def configure_extensions(self):
        config_info = self.find_mpi_config(self.mpi, self.mpi_cfg,
                                           MPICFG_ENV, MPICFG)
        if config_info:
            for ext in self.extensions:
                self.configure_extension(ext, config_info)
        return config_info

    def find_mpi_config(self, section, filenames,
                        envvars=None, defaults=None):
        # parse configuration file
        sect, fnames, cfg_info = _find_mpi_config(section, filenames,
                                                  envvars, defaults)
        if cfg_info:
            log.info("MPI configuration: "
                     "from section '%s' in file/s '%s'", sect, fnames)
        return cfg_info

    def configure_extension(self, extension, config_info):
        _configure(extension, config_info)


# --------------------------------------------------------

from distutils.core import Distribution as cls_Distribution
from distutils.extension import Extension
from distutils.core import Command

from distutils.command import install_lib as cmd_install_lib

# Distribution class supporting a 'executables' keyword

class Distribution(cls_Distribution):

    def __init__ (self, attrs=None):
        self.executables = None
        cls_Distribution.__init__(self, attrs)

    def has_executables(self):
        return self.executables and len(self.executables) > 0


# Executable class

class Executable(Extension):

    pass


# Command class to build executables

_build_ext = build_ext

class build_exe(_build_ext):

    description = "build binary executable components"

    user_options = [
        ('build-exe=', None,
         "build directory for executable components"),
        ] + _build_ext.user_options


    def initialize_options (self):
        _build_ext.initialize_options(self)
        self.build_base = None
        self.build_exe  = None

    def finalize_options (self):
        _build_ext.finalize_options(self)
        self.try_mpi_2 = None
        self.set_undefined_options('build',
                                   ('build_base','build_base'))
        from distutils.util import get_platform
        plat_specifier = ".%s-%s" % (get_platform(), sys.version[0:3])
        if self.build_exe is None:
            self.build_exe = os.path.join(self.build_base,
                                          'exe' + plat_specifier)
        # a bit of hack
        self.executables = self.distribution.executables
        self.extensions  = self.distribution.executables

    def get_exe_filename (self, exe_name):
        import string
        exe_path = string.split(exe_name, '.')
        # OS/2 has an 8 character module (extension) limit :-(
        if os.name == "os2":
            exe_path[len(exe_path) - 1] = exe_path[len(exe_path) - 1][:8]
        return os.path.join(*exe_path) + sysconfig.get_config_var('EXE')

    def get_ext_filename (self, ext_name):
        return self.get_exe_filename(ext_name)


    def build_extension (self, ext):
        from types import ListType, TupleType
        from distutils.dep_util import newer_group
        sources = ext.sources
        if sources is None or type(sources) not in (ListType, TupleType):
            raise DistutilsSetupError(
                ("in 'executable' option (extension '%s'), " +
                 "'sources' must be present and must be " +
                 "a list of source filenames") % ext.name
                )
        sources = list(sources)
        fullname = self.get_ext_fullname(ext.name)
        ext_filename = os.path.join(self.build_exe,
                                    self.get_ext_filename(fullname))
        depends = sources + ext.depends
        if not (self.force or newer_group(depends, ext_filename, 'newer')):
            log.debug("skipping '%s' executable (up-to-date)", ext.name)
            return
        else:
            log.info("building '%s' executable", ext.name)

        # Next, compile the source code to object files.

        # XXX not honouring 'define_macros' or 'undef_macros' -- the
        # CCompiler API needs to change to accommodate this, and I
        # want to do one thing at a time!

        # Two possible sources for extra compiler arguments:
        #   - 'extra_compile_args' in Extension object
        #   - CFLAGS environment variable (not particularly
        #     elegant, but people seem to expect it and I
        #     guess it's useful)
        # The environment variable should take precedence, and
        # any sensible compiler will give precedence to later
        # command line args.  Hence we combine them in order:
        extra_args = ext.extra_compile_args or []
        extra_args = extra_args[:]

        macros = ext.define_macros[:]
        for undef in ext.undef_macros:
            macros.append((undef,))

        objects = self.compiler.compile(sources,
                                        output_dir=self.build_temp,
                                        macros=macros,
                                        include_dirs=ext.include_dirs,
                                        debug=self.debug,
                                        extra_postargs=extra_args,
                                        depends=ext.depends)

        # XXX -- this is a Vile HACK!
        #
        # The setup.py script for Python on Unix needs to be able to
        # get this list so it can perform all the clean up needed to
        # avoid keeping object files around when cleaning out a failed
        # build of an extension module.  Since Distutils does not
        # track dependencies, we have to get rid of intermediates to
        # ensure all the intermediates will be properly re-built.
        #
        self._built_objects = objects[:]

        # Now link the object files together into a "shared object" --
        # of course, first we have to figure out all the other things
        # that go into the mix.
        if ext.extra_objects:
            objects.extend(ext.extra_objects)
        extra_args = ext.extra_link_args or []
        extra_args = extra_args[:]
        # Get special linker flags for building a executable with
        # bundled Python library, also fix location of needed
        # python.exp file on AIX
        ldshflag = sysconfig.get_config_var('LINKFORSHARED') or ''
        if sys.platform.startswith('aix'):
            python_lib = sysconfig.get_python_lib(standard_lib=1)
            python_exp = os.path.join(python_lib, 'config', 'python.exp')
            ldshflag = ldshflag.replace('Modules/python.exp', python_exp)
        extra_args.extend(ldshflag.split())
        # Detect target language, if not provided
        language = ext.language or self.compiler.detect_language(sources)
        self.compiler.link_executable(
            objects, ext_filename,
            output_dir=None,
            libraries=self.get_libraries(ext),
            library_dirs=ext.library_dirs,
            runtime_library_dirs=ext.runtime_library_dirs,
            extra_preargs=None,
            extra_postargs=extra_args,
            debug=self.debug,
            target_lang=language)


# Command class to install executables

class install_exe(cmd_install_lib.install_lib):

    description = "install binary executable components"

    def finalize_options (self):
        self.set_undefined_options('install',
                                   ('force', 'force'),
                                   ('skip_build', 'skip_build'))
        self.set_undefined_options('build_exe',
                                   ('build_exe', 'build_dir'))
        if os.name == 'posix':
            bindir = 'install_scripts'
        else:
            bindir = 'install_data'
        self.set_undefined_options('install',
                                   (bindir, 'install_dir'))

    def run (self):
        self.build()
        self.install()

    def build (self):
        if not self.skip_build:
            if self.distribution.has_executables():
                self.run_command('build_exe')

    def install (self):
        if os.path.isdir(self.build_dir):
            self.outfiles = self.copy_tree(self.build_dir, self.install_dir)
        else:
            self.warn("'%s' does not exist -- no executable to install" %
                      self.build_dir)
            self.outfiles = None

    def get_outputs (self):
        return self.outfiles


# Command class to clean executable

class clean_exe(Command):

    description = "clean up output of 'build_exe' command"

    user_options = [
        ('build-base=', 'b',
         "base build directory (default: 'build.build-base')"),
        ('build-exe=', None,
         "build directory for executable components "
         "(default: 'build_exe.build-exe')"),
        ('all', 'a',
         "remove all build output, not just temporary by-products")
        ]

    boolean_options = ['all']

    def initialize_options(self):
        self.build_base = None
        self.build_exe  = None
        self.all        = None

    def finalize_options(self):
        self.set_undefined_options('build',
                                   ('build_base', 'build_base'))
        self.set_undefined_options('build_exe',
                                   ('build_exe', 'build_exe'))
        self.set_undefined_options('clean',
                                   ('all', 'all'))

    def run (self):
        from distutils.dir_util import remove_tree
        if self.all:
            directory = self.build_exe
            if os.path.exists(directory):
                remove_tree(directory, dry_run=self.dry_run)
            else:
                log.warn("'%s' does not exist -- can't clean it",
                         directory)

# --------------------------------------------------------


