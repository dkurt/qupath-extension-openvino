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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import qupath.lib.common.Version;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.GitHubProject;
import qupath.lib.gui.extensions.QuPathExtension;
import qupath.opencv.dnn.DnnTools;
import qupath.opencv.ops.ImageOps;

/**
 * Extension to connect QuPath and Intel OpenVINO.
 *
 * @author Dmitry Kurtaev
 */
public class OpenVINOExtension implements QuPathExtension, GitHubProject {

	private final static Logger logger = LoggerFactory.getLogger(OpenVINOExtension.class);

	@Override
	public void installExtension(QuPathGUI qupath) {
		logger.debug("Installing OpenVINO extension");
		DnnTools.registerDnnModel(OpenVINODnnModel.class, OpenVINODnnModel.class.getSimpleName());
	}

	@Override
	public String getName() {
		return "OpenVINO extension";
	}

	@Override
	public String getDescription() {
		return "Add Intel OpenVINO support to QuPath";
	}

	@Override
	public GitHubRepo getRepository() {
		return GitHubRepo.create(getName(), "qupath", "qupath-extension-openvino");
	}

	@Override
	public Version getQuPathVersion() {
		return Version.parse("0.3.0-rc2");
	}


}
