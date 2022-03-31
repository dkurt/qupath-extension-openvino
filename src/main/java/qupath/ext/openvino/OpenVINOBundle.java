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

	private Core ie = new Core();
	private InferRequest[] requests;
	private int[] outShape;
	private int idx = 0;

	private String inpName;
	private String outName;

	public int tileHeight;
	public int tileWidth;

	private OpenVINOBundle(String pathModel, int tileHeight, int tileWidth) {
		logger.info("Initialize OpenVINO network for tile {}x{}", tileHeight, tileWidth);

		this.pathModel = pathModel;
		this.tileHeight = tileHeight;
		this.tileWidth = tileWidth;

		// Determine default number of async streams.
		Map<String, String> config = Map.of("CPU_THROUGHPUT_STREAMS", "CPU_THROUGHPUT_AUTO");
		ie.set_property("CPU", config);
		int nstreams = ie.get_property("CPU", "OPTIMAL_NUMBER_OF_INFER_REQUESTS").asInt();
		logger.info("Number of asynchronous streams: " + nstreams);

		String xmlPath = Paths.get(pathModel).toString();
		Model net = ie.read_model(xmlPath);

		// Get input and output info and perform network reshape in case of changed tile size
		inpName = net.input().get_any_name();
		int[] inpDims = net.input().get_shape();
		if (inpDims[1] != tileHeight || inpDims[2] != tileWidth) {
			inpDims[1] = tileHeight;
			inpDims[2] = tileWidth;
			net.reshape(inpDims);
		}

		outName = net.output().get_any_name();
		outShape = net.output().get_shape();

		// Initialize asynchronous requests.
		CompiledModel execNet = ie.compile_model(net, "CPU");

		requests = new InferRequest[nstreams];
		for (int i = 0; i < nstreams; ++i) {
			requests[i] = execNet.create_infer_request();
		}
	}


    private static Map<String, OpenVINOBundle> cachedBundles = new HashMap<>();

    static OpenVINOBundle loadBundle(String path, int tileHeight, int tileWidth) {
    	cachedBundles.clear();
		return cachedBundles.computeIfAbsent(path, p -> new OpenVINOBundle(p, tileHeight, tileWidth));
    }

    static OpenVINOBundle loadBundle(URI uri, int tileHeight, int tileWidth) {
    	return loadBundle(Paths.get(uri).toAbsolutePath().toString(), tileHeight, tileWidth);
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

		// Output blob shape is NHWC
		int outH = outShape[1];
		int outW = outShape[2];
		int outC = outShape[3];

		// We need to allocate a new output buffer for every run to avoid collisions.
		Mat outputMat = new Mat(outH, outW, opencv_core.CV_32FC(outC));
		Tensor output = OpenVINOTools.convertToBlob(outputMat);

		// Run inference
		Tensor input = OpenVINOTools.convertToBlob(inputs.get(inpName));
		synchronized (req) {
			req.set_input_tensor(input);
			req.set_output_tensor(output);
			req.start_async();
			req.wait_async();
			return Map.of(outName, outputMat);
		}
	}
}
