<p align="center">
  <h1 align="center">Dynamic Visual SLAM using a General 3D Prior</h1>



  <p align="center">
    <a href="https://github.com/PRBonn/Pi3MOS-SLAM"><img src="https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54" /></a>
    <a href="https://github.com/PRBonn/Pi3MOS-SLAM"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
    <img src="https://img.shields.io/badge/Paper-pdf-<COLOR>.svg?style=flat-square" /></a>
    <a href="https://lbesson.mit-license.org/"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" /></a>
  </p>


  <p align="center">
    <a href="https://www.ipb.uni-bonn.de/people/xingguang-zhong/index.html"><strong>Xingguang Zhong</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/people/liren-jin/index.html"><strong>Liren Jin</strong></a>
    ·
    <a href="https://www.tudelft.nl/en/staff/m.popovic/?cHash=07e8a5fb4eda6d511853b2bacaa92260"><strong>Marija Popović</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/people/jens-behley/"><strong>Jens Behley</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/people/cyrill-stachniss/"><strong>Cyrill Stachniss</strong></a>
  </p>
  <h3 align="center"> Paper</a> | Video</a></h3>
  <div align="center"></div>

![teaser](media/demo1.jpg)
![teaser](media/overview.jpg)

Tracking & Mapping | Map State Update |
:-: | :-: |
<video src='https://github.com/user-attachments/assets/8312aaf8-b530-464a-8b74-cc7c387eef22.mp4'> | <video src='https://github.com/user-attachments/assets/ace5c91d-bb4f-497f-b6e6-bb6daa6851e5.mp4'> |


<!-- TABLE OF CONTENTS -->

<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#abstract">Abstract</a>
    </li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#run">How to run it</a>
    </li>
    <li>
      <a href="#contact">Contact</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>


## Abstract

<details>
  <summary>[Details (click to expand)]</summary>
Reliable incremental estimation of camera poses and 3D reconstruction is key to enable various applications including robotics, interactive visualization, and augmented reality. However, this task is particularly challenging in dynamic natural environments, where scene dynamics can severely deteriorate camera pose estimation accuracy. In this work, we propose a novel monocular visual SLAM system that can robustly estimate camera poses in dynamic scenes. To this end, we leverage the complementary strengths of geometric patch-based online bundle adjustment and recent feed-forward reconstruction models. Specifically, we propose a feed-forward reconstruction model to precisely filter out dynamic regions, while also utilizing its depth prediction to enhance the robustness of the patch-based visual SLAM.
By aligning depth prediction with estimated patches from bundle adjustment, we robustly handle the inherent scale ambiguities of the batch-wise application of the feed-forward reconstruction model. Extensive experiments on multiple tasks show the superior performance of our proposed method compared to state-of-the-art approaches.
</details>

## Contact
If you have any questions, Feel free to contact:

- Xingguang Zhong {[zhong@igg.uni-bonn.de]()}
- Liren Jin {[ljin@uni-bonn.de]()}
