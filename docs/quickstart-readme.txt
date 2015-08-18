This script will setup a new conda environment with DistArray. Depending on the
prior availability of dependencies and system hardware, this process could take
anywhere from less than a minute upto a few hours. If installation fails,
delete the created conda environment [conda env remove -n <env-name>]
and re-run this script.

Prerequisites for using DistArray quickstart:
- A working conda installation (Anaconda/Miniconda)
- A working OSX package manager installation (MacPorts/HomeBrew)

This script allows you to configure some aspects of the resulting DistArray
install. Refer to the script usage message for details [./quickstart --help].
