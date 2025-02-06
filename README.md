# MyoSuite-MJX [WIP]

MyoSuite is a collection of musculoskeletal environments and tasks simulated with the [MuJoCo](https://github.com/google-deepmind/mujoco) physics engine and wrapped in the OpenAI `gym` API to enable the application of Machine Learning to bio-mechanic control problems.

This modified version integrates `MJX` for efficient execution.

---

## **Installation**

### **Prerequisites**
Ensure you have Python 3.8 or later installed.

It is recommended to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links) and create a separate environment:

```bash
conda create --name myosuite-mjx python=3.8
conda activate myosuite-mjx
```

### **Step 1: Install MuJoCo**

Clone and build MuJoCo:

```bash
git clone https://github.com/google-deepmind/mujoco.git
cd mujoco
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../install/ ..
cmake --build .
cmake --install .
```

### **Step 2: Build and Install MuJoCo Python Bindings**

```bash
cd ../python
bash make_sdist.sh
MUJOCO_PATH=/ABS_PATH/install MUJOCO_PLUGIN_PATH=/ABS_PATH/install/mujoco_plugin pip install dist/mujoco-3.2.8.tar.gz
```

Set the necessary environment variables:

```bash
export DYLD_LIBRARY_PATH=/ABS_PATH/install/lib:$DYLD_LIBRARY_PATH
```

### **Step 3: Install MJX**

```bash
cd ../mjx
pip install -e .
```

### **Step 4: Install MyoSuite**

```bash
pip install -e .
pip install jax mediapy
```

---

## **Running MyoSuite in MJX**

To test `MyoHand` in the modified environment, run:

```bash
python myosuite/mjx/play.py
```

---

## **License and Attribution**

This repository is based on [MyoSuite](https://github.com/myohub/myosuite) by Facebook AI and its original authors: Vikash Kumar, Vittorio Caggiano, and others. This modified version integrates MJX and ensures compatibility with the latest MuJoCo versions.

Original citation:

```BibTeX
@Misc{MyoSuite2022,
  author = {Vittorio, Caggiano AND Huawei, Wang AND Guillaume, Durandau AND Massimo, Sartori AND Vikash, Kumar},
  title = {MyoSuite -- A contact-rich simulation suite for musculoskeletal motor control},
  publisher = {arXiv},
  year = {2022},
  howpublished = {\url{https://github.com/myohub/myosuite}},
  doi = {10.48550/ARXIV.2205.13600},
  url = {https://arxiv.org/abs/2205.13600},
}
```

For more details, refer to the [original MyoSuite repository](https://github.com/myohub/myosuite).

