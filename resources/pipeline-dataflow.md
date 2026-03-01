```mermaid
flowchart TD
    HW["OV2311 Left and OV2311 Right - Global Shutter Sensors"]
    GMSL["GMSL2 Link - Arducam B0573 Serializer"]
    CSI["CSI-2 Port - /dev/video0"]
    GST_SRC["v4l2src - UYVY or NV16 input"]

    subgraph GST["GStreamer Pipeline — 3-tier fallback"]
        direction TB
        P1["Tier 1 — v4l2src NVMM UYVY - nvvidconv VIC - NV12 NVMM - appsink"]
        P2["Tier 2 — v4l2src NVMM UYVY - nvvidconv VIC - BGRx NVMM - appsink"]
        P3["Tier 3 — v4l2src UYVY - nvvidconv VIC - BGRx - appsink CPU"]
        P1 -->|"fallback"| P2
        P2 -->|"fallback"| P3
    end

    HW --> GMSL --> CSI --> GST_SRC --> GST

    GST --> CB["capture_loop thread - GstBuffer per frame"]

    CB --> DEC{"NVMM negotiated?"}

    subgraph NVBUF_PATH["NVBUF Hardware Path — requires HAVE_NVBUF"]
        direction TB
        NVMAP["gst_buffer_map NVMM - NvBufSurface ptr zero copy"]
        VIC1["VIC NvBufSurfTransformAsync x2 - NV12 crop left + right eye"]
        VIC2["VIC NvBufSurfTransformAsync x2 - BGRA crop left + right eye"]
        SYNC["Wait all 4 VIC sync objects - NvBufSurfTransformSyncObjWait"]
        CPU_MAP["NvBufSurfaceSyncForCpu - CPU cache flush all 4 surfaces"]
        NVMAP --> VIC1
        NVMAP --> VIC2
        VIC1 --> SYNC
        VIC2 --> SYNC
        SYNC --> CPU_MAP
    end

    subgraph CPU_PATH["CPU Fallback Path"]
        direction TB
        GMAP["gst_buffer_map CPU - pixel data DMA to RAM"]
        SPLIT["cv::Mat ROI split - left cols 0..W  right cols W..2W"]
        CVTC["cv::cvtColor BGRA to BGR - NEON vectorised"]
        GMAP --> SPLIT --> CVTC
    end

    DEC -->|"yes"| NVBUF_PATH
    DEC -->|"no"| CPU_PATH

    subgraph PUB["Publishers — left and right symmetric"]
        direction LR
        VS["visual_stream - image_raw or image/compressed - JPEG async"]
        NIT["nitros_image - NitrosImage TypeAdapter - GXF VideoBuffer"]
        CI["camera_info - sensor_msgs/CameraInfo - always"]
        TF["static TF - extrinsics relative to base_link"]
    end

    CPU_MAP --> PUB
    CVTC --> PUB

    NIT -->|"TypeAdapter convert_to_custom"| GXF["GXF VideoBuffer - lifecycle-managed - freed after consumption"]
```

## Step-by-Step Pipeline Explanation

1. **Hardware Capture**: Frames are captured by the OV2311 left and right global shutter sensors. They travel via the GMSL2 serial link to the CSI-2 port where they surface as a `/dev/video0` video node.
2. **GStreamer Ingestion & Conversion**: GStreamer uses `v4l2src` to ingest frames. A 3-tier fallback checks system capabilities to find the most efficient processing format:
   * **Tier 1 (Optimal)**: Uses NVMM memory (hardware-accelerated) and NV12 format.
   * **Tier 2 (Sub-optimal HW)**: Uses NVMM memory but BGRx format.
   * **Tier 3 (CPU Fallback)**: Uses CPU memory and BGRx format.
3. **Capture Loop**: The GStreamer `appsink` feeds `GstBuffer` frames into a continuous C++ capture loop thread.
4. **Buffer Processing (Zero-copy NVMM path)**: If hardware buffers (NVMM) are negotiated successfully:
   * The pipeline gets a zero-copy pointer (`NvBufSurface`).
   * The hardware Video Image Compositor (VIC) asynchronously crops the combined image into left and right camera images in both NV12 and BGRA formats (4 distinct operations).
   * CPU waits for all VIC sync objects to finish, then syncs the CPU cache so the data can be read by downstream publishers.
5. **Buffer Processing (CPU fallback path)**: If hardware buffers failed:
   * Memory is mapped to system RAM.
   * OpenCV (`cv::Mat`) Regions of Interest (ROI) are used to split the combined stereo frame in half.
   * `cv::cvtColor` uses ARM NEON vector instructions to quickly convert BGRA to BGR.
6. **Publishing**: Processed frames are sent to symmetric left and right ROS publishers:
   * `visual_stream`: Standard ROS `sensor_msgs/Image` and compressed JPEG streams.
   * `nitros_image`: High-speed zero-copy `NitrosImage` messages.
   * `camera_info` and `tf`: Camera calibration parameters and spatial extrinsics.
7. **Isaac ROS Integration**: When publishing `nitros_image`, the ROS `TypeAdapter` converts the data into NVIDIA's `GXF VideoBuffer`. This hardware-managed buffer is lifecycle-managed and provides zero-copy transport for integration with down-stream hardware-accelerated Isaac ROS nodes.