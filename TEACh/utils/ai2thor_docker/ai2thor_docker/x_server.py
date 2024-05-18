import subprocess
import time
import shlex
import re
import atexit
import platform
import tempfile
import threading
import os
import sys

# import ipdb
# st = ipdb.set_trace

def pci_records():
    records = []
    command = shlex.split('lspci -vmm')
    output = subprocess.check_output(command).decode()

    for devices in output.strip().split("\n\n"):
        record = {}
        records.append(record)
        for row in devices.split("\n"):
            key, value = row.split("\t")
            record[key.split(':')[0]] = value

    return records

def get_current_busid():
    result = subprocess.run(['nvidia-smi', '-a'], capture_output=True, text=True)
    nvidia_out = str(result.stdout) #.find('Bus Id')
    busid_ind = nvidia_out.find('Bus Id')
    busID_current_GPU = nvidia_out[busid_ind+45:busid_ind+47]
    busID_current_GPU_decimal = str(int(busID_current_GPU, 16))
    return busID_current_GPU_decimal

def generate_xorg_conf(devices):
    xorg_conf = []

    device_section = """
Section "Device"
    Identifier     "Device{device_id}"
    Driver         "nvidia"
    VendorName     "NVIDIA Corporation"
    BusID          "{bus_id}"
EndSection
"""
    server_layout_section = """
Section "ServerLayout"
    Identifier     "Layout0"
    {screen_records}
EndSection
"""
    screen_section = """
Section "Screen"
    Identifier     "Screen{screen_id}"
    Device         "Device{device_id}"
    DefaultDepth    24
    Option         "AllowEmptyInitialConfiguration" "True"
    SubSection     "Display"
        Depth       24
        Virtual 1250 1250
    EndSubSection
EndSection
"""
    screen_records = []
    for i, bus_id in enumerate(devices):
        xorg_conf.append(device_section.format(device_id=i, bus_id=bus_id))
        xorg_conf.append(screen_section.format(device_id=i, screen_id=i))
        screen_records.append('Screen {screen_id} "Screen{screen_id}" 0 0'.format(screen_id=i))
    
    xorg_conf.append(server_layout_section.format(screen_records="\n    ".join(screen_records)))

    output =  "\n".join(xorg_conf)
    return output

def _startx(busID_current_GPU_decimal, display_=None):
    # NOTE: DISPLAY IS NOT USED HERE


    if platform.system() != 'Linux':
        raise Exception("Can only run startx on linux")

    devices = []
    for r in pci_records():
        if r.get('Vendor', '') == 'NVIDIA Corporation'\
                and r['Class'] in ['VGA compatible controller', '3D controller']:
            bus_id = 'PCI:' + ':'.join(map(lambda x: str(int(x, 16)), re.split(r'[:\.]', r['Slot'])))
            devices.append(bus_id)
    # print(devices)

    if display_ is not None:
        display = display_
    else:
        # use busid of current gpu only
        # print(busID_current_GPU, busID_current_GPU_decimal)
        indices = [i for i, s in enumerate(devices) if busID_current_GPU_decimal in s]
        devices = [devices[i] for i in indices]
        print("Running xorg on these devices:", devices)

        # determine display number based on index position
        display = busID_current_GPU_decimal

    print("DISPLAY is ", display)

    if not devices:
        raise Exception("no nvidia cards found")

    try:
        fd, path = tempfile.mkstemp(dir='tmp')
        with open(path, "w") as f:
            f.write(generate_xorg_conf(devices))
        # print(os.path.split(path)[1])
        path = os.path.join('tmp',os.path.split(path)[1]) # alter path to be relative
        command = shlex.split("Xorg -noreset +extension GLX +extension RANDR +extension RENDER -config %s :%s" % (path, display))
        proc = subprocess.Popen(command)
        atexit.register(lambda: proc.poll() is None and proc.kill())
        proc.wait()
    finally: 
        os.close(fd)
        os.unlink(path)

def startx(display=None):
    # if 'DISPLAY' in os.environ:
    #     print("Skipping Xorg server - DISPLAY is already running at %s" % os.environ['DISPLAY'])
    #     return
    busID_current_GPU_decimal = get_current_busid() # this is used as display number
    print("DISP", display)

    xthread = threading.Thread(target=_startx, args=(busID_current_GPU_decimal,display))
    xthread.daemon = True
    xthread.start()
    # wait for server to start
    time.sleep(4)

    if display is None:
        disp_ = int(busID_current_GPU_decimal)
    else:
        disp_ = display

    return disp_


# if __name__ == "__main__":
#     startx(display=1)



    

