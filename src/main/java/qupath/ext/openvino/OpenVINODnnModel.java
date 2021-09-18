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

	// For every input shape there is a separate bundle.
	private transient List<OpenVINOBundle> bundles = new ArrayList<>();

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

	private OpenVINOBundle getBundle(int tileHeight, int tileWidth) {
		synchronized(bundles) {
			for (var b : bundles) {
				if (b.tileWidth == tileWidth && b.tileHeight == tileHeight) {
					return b;
				}
			}
			var bundle = OpenVINOBundle.loadBundle(uri, tileHeight, tileWidth);
			bundles.add(bundle);
			return bundle;
		}
	}

	class OVPredictionFunction implements PredictionFunction<Mat> {

		@Override
		public Map<String, Mat> predict(Map<String, Mat> inputs) {
			Mat input = (Mat)inputs.values().toArray()[0];
			return getBundle(input.rows(), input.cols()).run(inputs);
		}


		@Override
		public Mat predict(Mat input) {
			throw new java.lang.UnsupportedOperationException("predict is not implemented");
		}

		@Override
		public Map<String, DnnShape> getInputs() {
			throw new java.lang.UnsupportedOperationException("getInputs is not implemented");
		}

		@Override
		public Map<String, DnnShape> getOutputs(DnnShape... inputShapes) {
			throw new java.lang.UnsupportedOperationException("getOutputs is not implemented");
		}
	}

	static class OVBlobFun implements BlobFunction<Mat> {

		@Override
		public Mat toBlob(Mat... mats) {
			return mats[0];
		}

		@Override
		public List<Mat> fromBlob(Mat blob) {
			return List.of(blob);
		}

	}

}
