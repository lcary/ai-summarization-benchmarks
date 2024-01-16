# Troubleshooting

### CUDA Library Issues

If you see issues with torch or HF complaining about missing libraries, it's likely
an issue with the CUDA setup and the `ldconfig` configuration. Example error:
```
AssertionError: libcuda.so cannot found!
```

This may also appear as the following error:
```
nvrtc: error: failed to open libnvrtc-builtins.so.12.1.
  Make sure that libnvrtc-builtins.so.12.1 is installed correctly.
```

You will likely need to configure several settings for correct CUDA setup in Colab
by running the below commands:

```
!export CUDA_HOME=/usr/local/cuda-12.2
# Workaround: https://github.com/pytorch/pytorch/issues/107960
!ldconfig /usr/lib64-nvidia
!ldconfig -p | grep libcuda
```

The ldconfig command output should output libcuda libraries similar to the following list:
```
	libcudart.so.12 (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12
	libcudart.so (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so
	libcudadebugger.so.1 (libc6,x86-64) => /usr/lib64-nvidia/libcudadebugger.so.1
	libcuda.so.1 (libc6,x86-64) => /usr/lib64-nvidia/libcuda.so.1
	libcuda.so (libc6,x86-64) => /usr/lib64-nvidia/libcuda.so
```

If the ldconfig requires a different directory, check for other nvidia libraries
under /user/. If the notebook server has a different version of cuda home installed,
check for that via `ls /user/local/cuda*` and set that to CUDA_HOME. After that,
restart the session on the GPU server.

### Locale Encoding Issues

If you see an error in Colab like `NotImplementedError: A UTF-8 locale is required. Got ANSI_X3.4-1968`
you can fix this by entering the following into a cell and executing the block:
```
import locale
locale.getpreferredencoding = lambda: "UTF-8"
```
From https://github.com/googlecolab/colabtools/issues/3409#issuecomment-1446281277
