# QuPath OpenVINO extension

[![GitHub all releases](https://img.shields.io/github/downloads/dkurt/qupath-extension-openvino/total?color=blue)](https://github.com/dkurt/qupath-extension-openvino/releases)

Welcome to the Intel OpenVINO extension for [QuPath](http://qupath.github.io)!

This adds support for inference optimization using [Intel OpenVINO for Java](https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/java_api) into QuPath.


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
