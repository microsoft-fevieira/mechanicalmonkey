
This folder contains code from the
[Azure Kinect Python API(K4A)](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/tree/develop/src/python/k4a) repo.

[Installation instructions](https://docs.microsoft.com/en-us/azure/kinect-dk/sensor-sdk-download)

[Linux Device Setup](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md#linux-device-setup)

 To use the Azure Kinect SDK without being 'root', copy 'scripts/99-k4a.rules' into '/etc/udev/rules.d/'. Detach and reattach Azure Kinect devices.

Note:
If you get this this error:
[error] [t=4067] /__w/1/s/extern/Azure-Kinect-Sensor-SDK/src/dynlib/dynlib_linux.c (82): dynlib_create(). Failed to load shared object libdepthengine.so.2.0 with error: libdepthengine.so.2.0: cannot open shared object file: No such file or directory

check that libdepthengine.so.2.0 is among the results returned by:
sudo ldconfig -v

If it isn't:
sudo cp /usr/lib/x86_64-linux-gnu/libk4a1.4/libdepthengine.so.2.0  /usr/lib/x86_64-linux-gnu
sudo ldconfig