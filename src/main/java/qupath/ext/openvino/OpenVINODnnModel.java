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

import java.io.IOException;
import java.net.URI;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

import org.bytedeco.opencv.opencv_core.Mat;

import qupath.lib.io.UriResource;
import qupath.opencv.dnn.BlobFunction;
import qupath.opencv.dnn.DnnModel;
import qupath.opencv.dnn.DnnShape;
import qupath.opencv.dnn.PredictionFunction;

import org.intel.openvino.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class OpenVINODnnModel implements DnnModel<Mat>, UriResource {

	private final static Logger logger = LoggerFactory.getLogger(OpenVINODnnModel.class);

	private URI uri;

	private transient OpenVINOBundle bundle;

	private transient BlobFunction<Mat> blobFunction;
	private transient PredictionFunction<Mat> predictFunction;

	OpenVINODnnModel(URI uri) {
		this.uri = uri;
	}

	private PredictionFunction<Mat> createPredictionFunction() {
		return new OVPredictionFunction();
	}

	private BlobFunction<Mat> createBlobFunction() {
		return new OVBlobFun();
	}

	@Override
	public PredictionFunction<Mat> getPredictionFunction() {
		if (predictFunction == null) {
			synchronized(this) {
				if (predictFunction == null) {
					predictFunction = createPredictionFunction();
				}
			}
		}
		return predictFunction;
	}

	@Override
	public BlobFunction<Mat> getBlobFunction() {
		if (blobFunction == null) {
			synchronized(this) {
				if (blobFunction == null) {
					blobFunction = createBlobFunction();
				}
			}
		}
		return blobFunction;
	}

	@Override
	public BlobFunction<Mat> getBlobFunction(String name) {
		return getBlobFunction();
	}

	@Override
	public boolean updateUris(Map<URI, URI> replacements) throws IOException {
		var replace = replacements.getOrDefault(uri, null);
		if (replace != null && !Objects.equals(uri, replace)) {
			uri = replace;
			return true;
		}
		return false;
	}

	@Override
	public Collection<URI> getUris() throws IOException {
		return Collections.singletonList(uri);
	}

	private OpenVINOBundle getBundle() {
		if (bundle == null) {
			synchronized(this) {
				if (bundle == null) {
					logger.info("1");
					var bandle0 = OpenVINOBundle.loadBundle(uri);
					logger.info("2");
					bundle = bandle0;
				}
					logger.info("3");
			}
					logger.info("4");
		}
					logger.info("5");
		return bundle;
	}

	class OVPredictionFunction implements PredictionFunction<Mat> {

		@Override
		public Map<String, Mat> predict(Map<String, Mat> input) {
			// Blob input = OpenVINOTools.convertToBlob(mat);
			// System.out.println(input)
			logger.info("predict 2");
			var bundle = getBundle();
			logger.info("predict 2 - run");
			return bundle.run(input);
			// var function = getFunction();
			// return function.call(input);
		}


		@Override
		public Mat predict(Mat input) {
			logger.info("predict");
			return null;

			// var function = getFunction();
			// return function.call(input);
		}

		@Override
		public Map<String, DnnShape> getInputs() {
			logger.info("getInputs");

			return null;

			// return getBundle().getInputs().stream().collect(Collectors.toMap(
			// 		i -> i.getName(),
			// 		i -> DnnShape.of(i.getShape())
			// 		));
		}

		@Override
		public Map<String, DnnShape> getOutputs(DnnShape... inputShapes) {
			logger.info("getOutputs");

			return null;

			// return getBundle().getOutputs().stream().collect(Collectors.toMap(
			// 		i -> i.getName(),
			// 		i -> DnnShape.of(i.getShape())
			// 		));
		}
	}

	static class OVBlobFun implements BlobFunction<Mat> {

		@Override
		public Mat toBlob(Mat... mats) {
			return mats[0];
		}

		@Override
		public List<Mat> fromBlob(Mat blob) {
			List<Mat> out = new ArrayList<>();
			out.add(blob);
			logger.info("fromBlob");
			return out;
		}

	}

}
