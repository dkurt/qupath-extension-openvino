# QuPath OpenVINO extension

[![GitHub all releases](https://img.shields.io/github/downloads/dkurt/qupath-extension-openvino/total?color=blue)](https://github.com/dkurt/qupath-extension-openvino/releases) [![Awesome OpenVINO](https://img.shields.io/badge/Awesome-OpenVINO-FC60A8?logo=awesomelists)](https://github.com/dkurt/awesome-openvino)

Welcome to the Intel OpenVINO extension for [QuPath](http://qupath.github.io)!

This adds support for inference optimization using [Intel OpenVINO for Java](https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/java_api) into QuPath.

| Intel(R) Core(TM) i7-6700K | **Test image**: [OS-3.ndpi](https://cytomine.com/collection/os-3/os-3-ndpi)<br>**Model**: he_heavy_augment<br>**Tile size**: 1024x1024 |
|---|---|
| TensorFlow 2.4.1 with platform=mkl | 22:31 minutes |
| OpenVINO 2022.1 | 15:02 minutes (x1.48) |
| OpenVINO 2022.1 INT8 | 9:54 minutes (x2.33) |

<img src="https://user-images.githubusercontent.com/25801568/159727338-6284edae-5cd2-4536-80af-a26e67efe85b.png" width="256">

## Building

You can always build this extension from source but you can also download pre-built package from [releases](https://github.com/dkurt/qupath-extension-openvino/releases) page. Choose one for your operating system.

### Extension + dependencies separately

You can build the extension with

```bash
gradlew clean build copyDependencies
```

The output will be under `build/libs`.

* `clean` removes anything old
* `build` builds the QuPath extension as a *.jar* file and adds it to `libs`
* `copyDependencies` copies the TensorFlow dependencies to the `libs` folder

### Extension + dependencies together

Alternatively, you can create a single *.jar* file that contains both the
extension and all its dependencies with

```bash
gradlew clean shadowjar
```

## Installing

The extension + its dependencies will all need to be available to QuPath inside
QuPath's extensions folder.

The easiest way to install the jars is to simply drag them on top of QuPath
when it's running.
You will then be prompted to ask whether you want to copy them to the
appropriate folder.


## Usage

### OpenVINO IR format

OpenVINO uses own format for the deep learning networks representation (IR). It is a pair of `.xml` and `.bin` files which generated from original model. Download ready to use models from [models](./models) directory. There are FP32 and INT8 (quantized) version of the models. INT8 is faster for most of CPUs.

Alternatively, you can convert model locally. For model conversion you can install `openvino-dev` Python package and use Model Optimizer by `mo` command.

Example conversion for [StarDist](https://github.com/qupath/qupath-extension-stardist) model (we recommend to use Python virtual environment to install required packages):

```bash
python3 -m venv venv3
source venv3/bin/activate
pip install --upgrade pip
pip install openvino-dev tensorflow

mo --input input --data_type FP16 --input_shape "[1,1024,1024,3]" --saved_model_dir=he_heavy_augment
```

Note that extension is able to reshape model to any input size in runtime so `"[1,1024,1024,3]"` is just a default input resolution. For `dsb2018_heavy_augment` number of channels equals `1` so use `--input_shape "[1,1024,1024,1]"`
