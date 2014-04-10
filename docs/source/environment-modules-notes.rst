Notes on building environment-modules
=====================================

environment-modules is a tool, written with Tcl, that makes it convenient to
switch environment settings. It is not required to use distarray, but we find
it useful in development. It is a difficult name to google.  I had to build it
from source, and made some notes of my steps, which will hopefully be helpful
for others that build this.

There seems to be some version available to ``apt-get`` for Debian. But I read
suggestions not to mix Debian and Ubuntu packages, and as I have Ubuntu, I did
not try and configure my ``apt-get`` to look at the Debian packages. So I
installed from source, with notes as follows.

These specific notes are from an installation from source for Linux Mint
(Ubuntu), done by Mark Kness.  These actions were based on the INSTALL
document in the modules source and the Geoghegan link.

``$ sudo apt-get install tcl tcl8.4-dev``

This seemed to run ok.

``$ tar xvvf modules-3.2.10.tar.gz``

I had already downloaded this. Double v means extra verbose.

| ``$ cd modules-3.2.10``
| ``$ gedit README``
| ``$ gedit INSTALL``
| ``$ gedit INSTALL.RH7x``

Read the installation notes!

``$ ./configure``

First step is to run this and see how far it gets. Tcl is the likely problem here.

I got the following messages from ./configure...::

    checking for Tcl configuration (tclConfig.sh)... found /usr/lib/tcl8.4/tclConfig.sh
    checking for existence of tclConfig.sh... loading
    checking for Tcl version... 8.5
    checking TCL_VERSION... 8.5
    checking TCL_LIB_SPEC... -L/usr/lib -ltcl8.4
    checking TCL_INCLUDE_SPEC... -I/usr/include/tcl8.4
    checking for TclX configuration (tclxConfig.sh)... not found
    checking for TclX version... using 8.5
    checking TCLX_VERSION... 8.5
    checking TCLX_LIB_SPEC... TCLX_LIB_SPEC not found, need to use --with-tclx-lib
    checking TCLX_INCLUDE_SPEC... TCLX_INCLUDE_SPEC not found, need to use --with-tclx-inc
    configure: WARNING: will use MODULEPATH=/usr/local/Modules/modulefiles : rerun configure using --with-module-path to override default
    configure: WARNING: will use VERSIONPATH=/usr/local/Modules/versions : rerun configure using --with-version-path to override default

It seems that TCL_VERSION, TCL_LIB_SPEC, and TCL_INCLUDE_SPEC were all found
ok.  (The TCLX variants are not found but that is different and not a
problem.) Generally it seems like Tcl is ok, except perhaps for some 8.4 vs
8.5 version inconsistency.  A non-default path for the module files themselves
seems recommended, so...

| ``$ cd ~``
| ``$ mkdir modules``

This created ``/home/mkness/modules`` on my machine.  The install notes
suggest that one make a non-default location for these.  This directory name
was an arbitrary choice.

| ``$ cd modules-3.2.10``
| ``$ ./configure --with-module-path=~/modules``

Seemed ok. I ignored the version and prefix path options.

``$ make``

Seemed basically ok, a few warnings.

``$ ./modulecmd sh``

I got the usage instructions, and NOT any Tcl messages. Ok!

``$ sudo make install``

Seemed to run ok. Got permission errors without sudo.

| ``$ cd /usr/local/Modules``
| ``$ sudo ln -s 3.2.10 default``

Setup symbolic link named 'default' to point to the installed version.

| ``$ cd ~``
| ``$ /usr/local/Modules/default/bin/add.modules``

This script is supposed to update my local .bashrc and similar files to have
access to the Modules stuff.  For me, it modified .bashrc and .profile.  But
if I say 'module', I get an error about an invalid path.  It seems that
MODULE_VERSION is not defined, so I added ``export MODULE_VERSION=default`` to
the top of my .bashrc.

At this point I can say 'module' at the command line and I get the usage
instructions.  But 'module avail' dislikes the lack of an environment variable
MODULEPATH.  So I also add ``export MODULEPATH=~/modules`` to my .bashrc.
This path matches the --with-module-path argument to ./configure.

Now it works!

References
----------

http://modules.sourceforge.net/
The main page for the modules package.
It provides a source download: modules-3.2.10.tar.gz

http://sourceforge.net/p/modules/wiki/FAQ/
FAQ for the modules package.

http://nickgeoghegan.net/linux/installing-environment-modules
Build instructions for environment-modules. I partially followed these but with several changes.

http://packages.debian.org/wheezy/environment-modules
http://packages.debian.org/wheezy/amd64/environment-modules/download
http://packages.debian.org/unstable/main/environment-modules
Debian package for environment-modules. Note that this is two different places.

http://packages.debian.org/search?keywords=tcl&searchon=names&suite=stable&section=all
Debian package for Tcl.

