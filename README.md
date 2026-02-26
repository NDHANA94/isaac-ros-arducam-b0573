## arducam_dual_camera

A high-performance ROS 2 Humble package for the Arducam B0573 — a 2.3 MP Global Shutter Dual-Camera kit (GMSL2-to-CSI2) running on NVIDIA Jetson Orin Nano.

This node captures a side-by-side (SBS) stereo frame from the CSI-2 port, splits it into independent left and right image streams, and publishes them as standard sensor_msgs/Image or hardware-accelerated NitrosImage topics.

### Hardware Architecture

The B0573 uses two OV2311 global shutter sensors synchronized via a GMSL2 serializer/deserializer. The combined feed is delivered to the Jetson via a single CSI-2 interface.

```txt
[ Left Camera ]──┐
                  ├── GMSL2 link ──► Arducam B0573 GMSL2-to-CSI2 ──► CSI-2 Port (Jetson)
[ Right Camera ]─┘
```

**Sensor:** OV2311 (Global Shutter, 1600×1200 native resolution per eye)

**Interface:** GMSL2 → CSI-2 (/dev/video0)

**Platform:** NVIDIA Jetson Orin Nano / Xavier (JetPack 6.2+ / DeepStream 7.1+)



<!-- TODO: More descriptive explanation of the pipeline and dataflow, topics, etc -->