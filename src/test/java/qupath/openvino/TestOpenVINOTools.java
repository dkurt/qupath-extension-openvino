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

package qupath.openvino;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.junit.jupiter.api.Test;

import qupath.ext.openvino.OpenVINOTools;
import qupath.opencv.tools.OpenCVTools;

@SuppressWarnings("javadoc")
public class TestOpenVINOTools {

	@Test
	public void test_convertToBlob() {

		int rows = 4;
		int cols = 5;
		int channels = 3;
		Scalar values = new Scalar(1f, 2.5f, 4f, 0f);
		int n = 1;

		try (var scope = new PointerScope()) {
			var mat = new Mat(rows, cols, opencv_core.CV_32FC(channels), values);

			var tensor = OpenVINOTools.convertToBlob(mat);

			var shape = tensor.getTensorDesc().getDims();

			assertArrayEquals(shape, new int[] {n, channels, rows, cols});
			
			float[] data = new float[tensor.size()];
			tensor.rmap().get(data);

			// Check values are correct
			int idx = 0;
			for (long b = 0; b < n; b++) {
				for (long r = 0; r < rows; r++) {
					for (long c = 0; c < cols; c++) {
						for (long channel = 0; channel < channels; channel++) {
							float v = data[idx];
							assertEquals(v, values.get(channel), 1e-6);
							idx += 1;
						}
					}
				}
			}
		}

	}

}
