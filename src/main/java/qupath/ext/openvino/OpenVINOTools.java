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

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.common.GeneralTools;
import qupath.lib.regions.Padding;
import qupath.opencv.dnn.DnnModel;
import qupath.opencv.ops.ImageOp;
import qupath.opencv.tools.OpenCVTools;

import org.intel.openvino.*;

/**
 * Helper methods for working with Intel OpenVINO and QuPath, with the help of OpenCV.
 *
 * @author Dmitry Kurtaev
 */
public class OpenVINOTools {

    private final static Logger logger = LoggerFactory.getLogger(OpenVINOTools.class);

	static {
        Core.loadNativeLibs();
	}

    /**
     * Wrap OpenCV Mat to OpenVINO Blob to pass it then to the network.
     *
     * @param mat OpenCV Mat which represents an image with interleaved channels order
     * @return OpenVINO Blob
     */
    public static Tensor convertToBlob(Mat mat) {
        int[] dimsArr = {1, mat.rows(), mat.cols(), mat.channels()};
		return new Tensor(ElementType.f32, dimsArr, mat.data().address());
    }

	/**
	 * Create a {@link DnnModel} for OpenVINO by reading a specified model file.
	 * @param modelPath
	 * @return
	 * @throws IOException
	 */
	public static DnnModel<Mat> createDnnModel(String modelPath) throws IOException {
		try {
			return createDnnModel(GeneralTools.toURI(modelPath));
		} catch (URISyntaxException e) {
			throw new IOException(e);
		}
	}

	/**
	 * Create a {@link DnnModel} for OpenVINO by reading a specified URI.
	 * @param uri
	 * @return
	 */
	public static DnnModel<Mat> createDnnModel(URI uri) {
		return new OpenVINODnnModel(uri);
	}


}
