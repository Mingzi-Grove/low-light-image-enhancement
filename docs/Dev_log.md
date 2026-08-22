## colorspace.py        | 2026.7.30

### **关于取值范围与归一化**

BGR -> LAB 是 cv2 做的，输出是 numpy.ndarray float32 [L(0-100), A(-128-127), B(-128-127)]，而 LAB -> LCH 需要自己写（cv2）做不到。

**point** 所以LCH 空间的数值范围可以自己设定，服务于后续的降噪/增益等数学计算，取值范围需要方便理解和计算。

- **关于降噪**
    - 相比于 XYZ，LAB space 解耦出独立的亮度信息 L，但 Chroma 和 Hue 还耦合在 a-b 平面里，LCH 能解耦 Lightness、Chroma 和 Hue。我尚未了解低光照图像的噪声分布，不知道噪声是否也可以解耦到这三个部分并分别进行降噪处理。
    - 分析噪声的产生过程 物理光线 -> 光学镜头 -> 光电传感器 -> 存储序列，噪声的产生过程并不是按 LCH 空间的三个维度产生的，所以降噪应该放在哪里？哪里的降噪逻辑最顺？哪里的降噪效果最好？
- **关于增益/补偿**
    - 亮度增益对 L 操作
    - 色彩补偿对 C 操作
    - 既然增益/补偿的操作倾向于在 LCH 空间进行，而降噪的位置/方法还没确定，那就先试试让 LCH 的取值范围服务于增益/补偿的数学运算？

## chorma_comp.py       | 2026.7.30

### database
这些数据集值得寻找，不过目前只成功下载到 LOL 数据集
- MIT-Adobe FiveK
- LOw-Light paired dataset
- Vision Enhancement in Low-Light
- LENVIz
- MILL

## colorspace.py        | 2026.7.31

**关于边界处理** 我意识到了边界处理的问题，目前的代码中还没有做边界处理。目前小项目还在开发的早期，我感觉现在做好边界处理可能会掩盖后续增益超限的问题。我想留意这个地方，等后面写增益算法和测试时需要用到哪些边界检测再回来写。

## lightness_gain.py    | 2026.7.31

**规划的目标 Retinex 增强算法**
**初步尝试增益算法** 但现在先尝试 CLAHE 和 Gamma 这些简单的增益算法，先跑通项目

## lightness.py         | 2026.8.1

**ΔL** 我意识到 chroma 的补偿需要 ΔL 驱动，ΔL 是 L_lch 的增量，返回 ΔL 的功能应该写在哪个模块里？lightness 还是 enhance？我发现自己理不清 enhance 和 lightness, chroma 这两层封装。需要思考...
- **think**
    - chroma.compensate() 的 args 应该是 ΔL, C_lch, method
    - enhance 中 L_lch_gained = lightness.gain(L_lch) 后可以自然地拿到增益前后的 L，相减就能得到 ΔL，但这个过程在 lightness 内应该也方便得到...
- **接口的设计逻辑**
    - 一个函数的接口，应该由"它自己天然能产出什么"决定，而不是由"下游想要什么"决定。
    - 这样上游的输出就不会被“绑定”在下游，下游函数改动时，上游的函数就不需要变动。
    - 所以 **ΔL 自然由 lightness 模块产生**，需要考虑怎么让 lightness 给出返回多个结果的接口
- **模块接口返回多个结果**
    - **way 1. 模块级共享变量（黑板）** 简单直观，但有 重入 的问题。
        - **重入** 上一次调用还没消费完结果，下一次调用又把数据写进同一块共享内存，挤掉了正要用到的数据。
            - **注意** 函数内局部变量每次调用各有独立内存，反而没有这个问题。
    - **way 2. 结构体** 声明一个结构体，之后每次调用函数都会有一个专门对应的结构体（包）来装输出，不需要放在别的地方
        - **3 种相关的数据类型**
            - **tuple** 没名字，只能靠 tuple[0] tuple[1] 访问；不可变，写完了就不能改
            - **NamedTuple** 有元素名的 tuple，可以用元素的名字访问 result.L_gained；不可变
            - **dataclass** 有元素名；可变；可挂方法（更自由）

## chroma.py            | 2026.8.1

用线性增益占位跑通了饱和度补偿，但发现了问题：
在补偿没有越界的情况下，补偿后的图像出现了条带状的色带，具体原因还没确定，打算从 C、H 这两层的值和变化结合 LCH 空间的转换函数分析
待验证：① 条带区域原始 C 是否≈0？② 原始 H 是否噪声大？③ 实验：只对 C>阈值 的像素补偿，灰色区不动，条带是否消失？

## pipeline.py           | 2026.8.2

Pipeline 包括整个处理流程
- **重命名** 我意识到之前的 enhance 的模块命名是动词，不符合约定，于是把它改为 pipeline。
- **TODO** 可选参数

## cli.py               | 2026.8.2
开发了命令行入口，功能很简单:
>python -m input_path --output_dir examples\output
然后就会在指定的路径下保存处理后的 image
- **Learn**
    - **import argparse** Python 的一个标准库，用于管理参数
        - 可选参数的的设置
        > parser.add_argument("--output_dir", nargs="?", default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
            - **nargs** number of arguments，告诉 argparse 这个参数需要输入几个值。当前的 "?" 表示跟 0 或 1 个值，可有可无，没有就用默认值，宽松一些。
    - **from pathlib import Path** 一个解析路径的功能
        >path = Path(args.input_path)
        - **path（路径）** 一串**地址**，描述位置，指向 directory 或 file 这两个**实体**
            - **directory（目录）** 某个**文件夹**（实体）
            >path.is_dir()  # 是否指向一个 directory
            - **file（文件）** 某个**具体的文件**（实体）
            >path.is_file() # 是否指向一个 file
        - **path.stem**：路径的最后一截，不带后缀的文件名
        - **path.suffix**：路径最后一截的文件名后缀（例如“.png”）
    - **print(f"xxx", file=sys.stderr)，其中的“stderr”是什么？** 
    一个程序有两个输出通道，stdout（标准输出） 和 stderr（标准错误），前者经常被重定向输出到某个 log.txt 文件，例如
    >python xxx.py > log.txt    # 把 stdout 存进文件
    但后者（错误输出）需要立即显示到屏幕上。

## matrics.py           | 2026.8.3
- **学习几个评价指标**
    - **PSNR**
    - **SSIM**
    - **LOE**
    - **NIQE**
- **Coding**
    -**输入的 dtype 应该是什么** float32

## 数据集               | 2026.8.5
我需要几个数据集，包括
1. 低光光照与正常光照图像的数据集
- **LOw-Light dataset, LOL** ok
2. 低光照图像的降噪数据集（干净的参考图像与含有噪声的图像）
- **RENOIR** ok

## matrics.py           | 2026.8.6
现在可以通过 matrics.ssim() 计算 ssim 指标了，和 skimage 的结果差距在小数点后第三位，是可以用的。
不过我发现 SSIM 的窗口大小与图像分辨率的比例没有明确的要求，这可能导致不同尺寸的图像以不同窗长计算出的 SSIM 有明显差距。当窗口大小与**人眼能感知到结构区域大小**匹配时，SSIM 的参考意义才更大。
而图像本身就是多尺度的（例如有远处的大片的山脉，也有近处的小树叶纹理），，单一的窗口大小很难同时概括图像中天然的多尺度结构，所以**MS-SSIM**出现了。


## denosier.py          | 2026.8.6
了解了一部分降噪方法
实现了高斯模糊
实现了双边滤波
实现了 NLM，单通道和 bgr 都可以，NLM 似乎适合前置降噪
测试并对比了这三种降噪方法，从 matrics 和主观视觉

## pipeline.py          | 2026.8.7
加入了前置的 NLM 降噪，直接处理 bgr u8 的图像

## tests/               | 2026.8.7

### 数据转换的测试      | 2026.8.7

项目中图像转换是这样实现的:
>**BGR u8** --utils.nomalize--> **BGR f32** --cv2.cvtColor--> **LAB f32** --utils.LAB2LCH--> **LCH f32**--utils.LCH2LAB -->**LAB f32** --cv2.cvtColor--> **BGR f32** --utils.denomalize--> **BGR u8**

不光要测试每步处理后的 dtype 和 region，还要注意转换的误差。整个流程中转换有 3 层,需要测试每一层的误差大小，才能定位误差的来源。
>1. BGR_u8 -> BGR f32 -> BGR_u8
>2. BGR_f32 -> LAB_f32 -> BGR_f32
>3. LAB_f32 -> LCH_f32 -> LAB_f32

测试结果表明转换链误差的主要来源是 cv2.cvtColor，在 bgr f32 <-> lab f32 最大误差有 0.0037，平均误差 0.0009，理论上最大误差要小于 0.0019 才能实现 bgr u8 无损，目前的误差造成 bgr u8 一个像素级的误差。

## log                  | 2026.8.8
用 logger. 替换了模块中的 print。

## README.md            | 2026.8.8
修改了 README 文件，写得更清楚了。

## .gitignore           | 2026.8.9
重写 .gitignore，做了合适的筛选。

## docs/releases/        | 2026.8.9
新建了这个文件夹，用于存放版本说明文档，现在已经有 `v0.0.1.md` 在里面了。

## 项目结构调整          | 2026.8.9
为了方便以后添加一些新的“草稿笔记”，我在 `notebooks/` 下新建了一个 `/scratch` 文件夹，我在里面新建了 `git.md` ，用于记录学习 git 的笔记，还把原来的 `flowtest.ipynb` 和 `learn-pytest/` 一同放进去了，这样在 .gitignore 里就只需要写 /notebooks/scratch/，这样就不需要挨个枚举草稿笔记了。`.gitignore` 已做同步更新。

## 添加亮度增益算法      | 2026.8.11
- 新增 `algorithms/histogram.py`，源代码，包括 HE 和 CLAHE 两个方法
- 更新 `algorithms/__init__.py`，新增声明 `from . import histogram` 
- 修改 `lightness.boost()` 和 `gamma()`，参数调整
    - `boost(channel, gamma: float = 0.5) -> boost(channel)`
    - `gamma(img_gray) -> gamma(channel, exponent: float = 0.5)`
- 更新 `notebooks/Algorithms/Histogram/`，相关学习笔记
- 更新 `Dev_log.md`

## 开始分析数据集       | 2026.8.13
尝试写新的亮度增益算法时遇到了“but wait moment”，亮度增益的目标是输出亮度信息与真实正常光照环境下相近的图像，需要研究低光照图像与正常光照图像的亮度差异。

## 添加 debug 文档      | 2026.8.18
在 `docs/` 下添加 `debug_notes/` 文件夹，记录问题和 debug 过程。

## 分析数据集           | 2026.8.22
[analysis.ipynb](../data/lol/analysis.ipynb) 对比 LOL 数据集中亮暗样本的 Lightness 差异，寻找亮度增益的规律
