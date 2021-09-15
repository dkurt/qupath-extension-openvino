/*-
 * #%L
 * This file is part of QuPath.
 * %%
 * Copyright (C) 2021 QuPath developers, The University of Edinburgh
 * %%
 * QuPath is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * QuPath is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QuPath.  If not, see <https://www.gnu.org/licenses/>.
 * #L%
 */

package qupath.ext.openvino;

import java.io.File;
import java.net.URI;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.LongStream;
import java.util.ArrayList;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.opencv_core.Mat;

import org.intel.openvino.*;

class OpenVINOBundle {

	private final static Logger logger = LoggerFactory.getLogger(OpenVINOBundle.class);

	private String pathModel;

	private IECore ie = new IECore();
	private InferRequest[] requests;
	private int[] outShape;
	private int idx = 0;

	private String inpName;
	private String outName;
	// private List<SimpleTensorInfo> inputs;
	// private List<SimpleTensorInfo> outputs;

	// private MetaGraphDef metaGraphDef;

	private OpenVINOBundle(String pathModel) {
		logger.info("Initialize OpenVINO network");

		this.pathModel = pathModel;

		// Determine default number of async streams.
		Map<String, String> config = Map.of("CPU_THROUGHPUT_STREAMS", "1");
		// Map<String, String> config = Map.of("CPU_THROUGHPUT_STREAMS", "CPU_THROUGHPUT_AUTO");
		ie.SetConfig(config, "CPU");
		String nStr = ie.GetConfig("CPU", "CPU_THROUGHPUT_STREAMS").asString();
		int nstreams = Integer.parseInt(nStr);
		logger.info("Number of asynchronous streams: " + nstreams);

		String xmlPath = Paths.get(pathModel).toString();
		CNNNetwork net = ie.ReadNetwork(xmlPath);

		// Get input and output info and perform network reshape in case of changed tile size
		Map<String, InputInfo> inputsInfo = net.getInputsInfo();
		inpName = new ArrayList<String>(inputsInfo.keySet()).get(0);
		InputInfo inputInfo = inputsInfo.get(inpName);

		Map<String, Data> outputsInfo = net.getOutputsInfo();
		outName = new ArrayList<String>(outputsInfo.keySet()).get(0);
		Data outputInfo = outputsInfo.get(outName);

		outShape = outputInfo.getDims();

		// int[] inpDims = inputInfo.getTensorDesc().getDims();
		// if (inpDims[2] != tileHeight || inpDims[3] != tileWidth) {
		// 	inpDims[2] = tileHeight;
		// 	inpDims[3] = tileWidth;
		// 	Map<String, int[]> shapes = new HashMap<>();
		// 	shapes.put(inpName, inpDims);
		// 	net.reshape(shapes);
		// }
		inputInfo.setLayout(Layout.NHWC);
		outputInfo.setLayout(Layout.NHWC);
		ExecutableNetwork execNet = ie.LoadNetwork(net, "CPU");

		requests = new InferRequest[nstreams];
		for (int i = 0; i < nstreams; ++i) {
			requests[i] = execNet.CreateInferRequest();
		}
	}


    private static Map<String, OpenVINOBundle> cachedBundles = new HashMap<>();

    static OpenVINOBundle loadBundle(String path) {
    	cachedBundles.clear();
    	return cachedBundles.computeIfAbsent(path, p -> new OpenVINOBundle(p));
    }

    static OpenVINOBundle loadBundle(URI uri) {
    	return loadBundle(Paths.get(uri).toAbsolutePath().toString());
    }

	/**
	 * Get the path to the model (an .xml file).
	 * @return
	 */
	public String getModelPath() {
		return pathModel;
	}

	public Map<String, Mat> run(Map<String, Mat> inputs) {
		InferRequest req = null;
		synchronized (requests) {
			req = requests[idx];
			idx = (idx + 1) % requests.length;
		}

		// Output blob shape is NCHW. However output data layout is in NHWC (see outputInfo.setLayout).
		int outC = outShape[1];
		int outH = outShape[2];
		int outW = outShape[3];
		// We need to allocate a new output buffer for every run to avoid collisions.
		Mat outputMat = new Mat(outH, outW, opencv_core.CV_32FC(outC));

		TensorDesc tDesc = new TensorDesc(Precision.FP32, outShape, Layout.NHWC);
		Blob output = new Blob(tDesc, outputMat.data().address());

		// Run inference
		Blob input = OpenVINOTools.convertToBlob(inputs.get(inpName));
		synchronized (req) {
			req.SetBlob(outName, output);
			req.SetBlob(inpName, input);
			req.StartAsync();
			req.Wait(WaitMode.RESULT_READY);
			return Map.of(outName, outputMat);
		}
	}

	// public long[] getOutputShape(String name) {
	// 	var op = bundle.graph().operation(name);
	// 	if (op == null)
	// 		return null;
	// 	int nOutputs = op.numOutputs();
	// 	if (nOutputs > 1) {
	// 		logger.warn("Operation {} has {} outputs!", name, nOutputs);
	// 	} else if (nOutputs == 0)
	// 		return new long[0];
	// 	var shapeObject = op.output(0).shape();
	// 	long[] shape = new long[shapeObject.numDimensions()];
	// 	for (int i = 0; i < shape.length; i++)
	// 		shape[i] = shapeObject.size(i);
	// 	return shape;
	// }

	/**
	 * Get information about the first required output (often the only one).
	 * @return
	 */
	// public SimpleTensorInfo getInput() {
	// 	return inputs == null || inputs.isEmpty() ? null : inputs.get(0);
	// }

	/**
	 * Get information about all required inputs, or an empty list if no information is available.
	 * @return
	 */
	// public List<SimpleTensorInfo> getInputs() {
	// 	return inputs == null ? Collections.emptyList() : Collections.unmodifiableList(inputs);
	// }

	/**
	 * Get the first provided output (often the only one).
	 * @return
	 */
	// public SimpleTensorInfo getOutput() {
	// 	return outputs == null || outputs.isEmpty() ? null : outputs.get(0);
	// }

	/**
	 * Get information about all provided outputs, or an empty list if no information is available.
	 * @return
	 */
	// public List<SimpleTensorInfo> getOutputs() {
	// 	return outputs == null ? Collections.emptyList() : Collections.unmodifiableList(outputs);
	// }

	/**
	 * Returns true if the model takes a single input.
	 * @return
	 */
	// public boolean singleInput() {
	// 	return inputs != null && inputs.size() == 1;
	// }

	/**
	 * Returns true if the model provides a single output.
	 * @return
	 */
	// public boolean singleOutput() {
	// 	return outputs != null && outputs.size() == 1;
	// }

	// @Override
	// public String toString() {
	// 	if (singleInput() && singleOutput())
	// 		return String.format("TensorFlow bundle: %s, (input=%s, output=%s)",
	// 				getModelPath(), getInput(), getOutput());
	// 	return String.format("TensorFlow bundle: %s, (inputs=%s, outputs=%s)",
	// 			getModelPath(), getInputs(), getOutputs());
	// 	//    	return String.format("TensorFlow bundle: %s, (input%s [%s], output=%s [%s])",
	// 	//    			pathModel, inputName, arrayToString(inputShape), outputName, arrayToString(outputShape));
	// }

	/**
	 * Helper class for parsing the essential info for an input/output tensor.
	 */
	// public static class SimpleTensorInfo {

	// 	private TensorInfo info;
	// 	private String name;
	// 	private long[] shape;

	// 	SimpleTensorInfo(String name, TensorInfo info) {
	// 		this.info = info;
	// 		this.name = name;
	// 		if (info.hasTensorShape()) {
	// 			var dims = info.getTensorShape().getDimList();
	// 			shape = new long[dims.size()];
	// 			for (int i = 0; i < dims.size(); i++) {
	// 				var d = dims.get(i);
	// 				shape[i] = d.getSize();
	// 			}
	// 		}
	// 	}

	// 	TensorInfo getInfo() {
	// 		return info;
	// 	}

	// 	/**
	// 	 * Get any name associated with the tensor.
	// 	 * @return
	// 	 */
	// 	public String getName() {
	// 		return name;
	// 	}

	// 	/**
	// 	 * Get the tensor shape as an array of long.
	// 	 * @return
	// 	 */
	// 	public long[] getShape() {
	// 		return shape == null ? null : shape.clone();
	// 	}

	// 	@Override
	// 	public String toString() {
	// 		if (shape == null) {
	// 			return name + " (no shape)";
	// 		} else {
	// 			return name + " (" +
	// 					LongStream.of(shape).mapToObj(l -> Long.toString(l)).collect(Collectors.joining(", "))
	// 					+ ")";
	// 		}
	// 	}

	// }


}
