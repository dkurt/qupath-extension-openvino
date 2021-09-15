/*-
 * #%L
 * This file is part of QuPath.
 * %%
 * Copyright (C) 2020 - 2021 QuPath developers, The University of Edinburgh
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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.images.servers.ImageChannel;
import qupath.lib.regions.Padding;
import qupath.opencv.ops.ImageOp;
import qupath.opencv.ops.ImageOps.PaddedOp;
import qupath.opencv.tools.OpenCVTools;

import org.intel.openvino.*;

/**
 * An {@link ImageOp} that runs a TensorFlow model for prediction.
 *
 * @author Pete Bankhead
 * @deprecated use instead {@link TensorFlowDnnModel}
 */
@Deprecated
class OpenVINOOp extends PaddedOp {

	private final static Logger logger = LoggerFactory.getLogger(OpenVINOOp.class);

	private final static int DEFAULT_TILE_SIZE = 512;

	private String modelPath;
	private int tileWidth = 512;
	private int tileHeight = 512;

	private Padding padding;

	// Identifier for the requested output node - may be null to use the default output
	private String outputName = null;

	private transient OpenVINOBundle bundle;
	private transient Exception exception;

	OpenVINOOp(String modelPath, int tileWidth, int tileHeight, Padding padding, String outputName) {
		super();
		logger.debug("Creating op from {}", modelPath);
		this.modelPath = modelPath;
		this.outputName = outputName;
		this.tileWidth = tileWidth;
		this.tileHeight = tileHeight;
		if (padding == null)
			this.padding = Padding.empty();
		else
			this.padding = padding;

        IECore.loadNativeLibs();
	}

	private OpenVINOBundle getBundle() {
		if (bundle == null && exception == null) {
			try {
				bundle = OpenVINOBundle.loadBundle(modelPath);
			} catch (Exception e) {
				logger.error("Unable to load bundle: " + e.getLocalizedMessage(), e);
				this.exception = e;
			}
		}
		return bundle;
	}

	// Not needed
	@Override
	protected Padding calculatePadding() {
		return padding;
	}

	@Override
	protected Mat transformPadded(Mat input) {
		logger.info("transformPadded");
		var bundle = getBundle();
		if (exception != null)
			throw new RuntimeException(exception);

		try (@SuppressWarnings("unchecked")
		var scope = new PointerScope()) {
			Mat output;

			// var function = bundle.getFunction();
			// var signature = function.signature();
			// String inputName = signature.inputNames().iterator().next();
			// String outputName = signature.outputNames().iterator().next();

			// if (tileWidth > 0 && tileHeight > 0)
			// 	output = OpenCVTools.applyTiled(m -> run(function, m, inputName, outputName), input, tileWidth, tileHeight, opencv_core.BORDER_REFLECT);
			// else
			// 	output = run(function, input, inputName, outputName);

			// input.put(output);
		}
		return input;
	}

	// private static Mat run(ConcreteFunction function, Mat mat, String inputName, String outputName) {

	// 	var tensor = TensorFlowTools.convertToTensor(mat);
	// 	var outputMap = function.call(Map.of(inputName, tensor));
	// 	var outputTensor = outputMap.get(outputName);

	// 	var output = TensorFlowTools.convertToMat(outputTensor);

	// 	for (var val : outputMap.values())
	// 		val.close();
	// 	tensor.close();

	// 	return output;
	// }



	@Override
	public Padding getPadding() {
		return super.getPadding();
//        return Padding.empty();
    }

//     @Override
//    public List<ImageChannel> getChannels(List<ImageChannel> channels) {
//         var names = new ArrayList<String>();
//         var bundle = getBundle();
//         var output = bundle.getOutput();
//         long[] shape;
//         String name = outputName;
//         if (outputName == null || outputName.equals(output.getName())) {
//         	name = output.getName();
//         	shape = output.getShape();
//         } else
//         	shape = bundle.getOutputShape(outputName);
//         if (shape == null) {
//         	logger.warn("Cannot determine number of output channels - output shape is unknown!");
//         	return channels;
//         }
//         var nChannels = shape[shape.length-1];
//         for (int i = 0; i < nChannels; i++)
//             names.add(name + " " + i);
//         return ImageChannel.getChannelList(names.toArray(String[]::new));
//     }


}
