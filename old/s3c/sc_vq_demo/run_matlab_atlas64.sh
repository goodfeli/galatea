
   #!/bin/sh
   # please set this variable to the path of your gcc run-time library
   export LIB_GCC=/usr/lib/gcc/x86_64-redhat-linux/4.5.1/
   # if you can not use the mex files, uncomment this line.
   export LD_PRELOAD=$LIB_GCC/libgcc_s.so:$LIB_GCC/libstdc++.so:$LIB_GCC/libgomp.so 
   export LD_LIBRARY_PATH=/u/goodfeli/SPAMS/libs_ext/atlas64/
   # if your matlab crashes, please try to uncomment the following line
   # export LD_PRELOAD=$LIB_GCC/libgomp.so 
   matlab $* -r "addpath('/u/goodfeli/SPAMS/release/atlas64/'); addpath('/u/goodfeli/SPAMS/test_release'); "
