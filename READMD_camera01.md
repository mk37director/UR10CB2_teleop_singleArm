[sudo] password for ur12e: 
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
v4l-utils is already the newest version (1.22.1-2build1).
0 upgraded, 0 newly installed, 0 to remove and 27 not upgraded.
ur12e@ur10-cb2:~$ v4l2-ctl --list-devices
C922 Pro Stream Webcam (usb-0000:07:00.1-3):
	/dev/video2
	/dev/video3
	/dev/media1

C922 Pro Stream Webcam (usb-0000:07:00.1-4):
	/dev/video0
	/dev/video1
	/dev/media0




ros2 run v4l2_camera v4l2_camera_node --ros-args   -p video_device:="/dev/video0"   -r __ns:="/left"   -r __node:="camera"



ros2 run v4l2_camera v4l2_camera_node --ros-args   -p video_device:="/dev/video2"   -r __ns:="/right"   -r __node:="camera"




ros2 run rqt_gui rqt_gui

