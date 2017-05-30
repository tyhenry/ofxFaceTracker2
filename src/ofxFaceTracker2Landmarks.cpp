#include "ofxFaceTracker2Landmarks.h"


ofxFaceTracker2Landmarks::ofxFaceTracker2Landmarks(dlib::full_object_detection shape, ofxFaceTracker2InputInfo & info) : shape(shape), info(info){
}




ofVec2f ofxFaceTracker2Landmarks::getImagePoint(int i) const {
    ofVec3f p = ofVec3f(shape.part(i).x(),
                        shape.part(i).y(), 0);
    p = p * info.rotationMatrix;
    
    return ofVec2f(p);
}

vector<ofVec2f> ofxFaceTracker2Landmarks::getImagePoints() const {
    int n = shape.num_parts();
    vector<ofVec2f> imagePoints(n);
    for(int i = 0; i < n; i++) {
        imagePoints[i] = getImagePoint(i);
    }
    return imagePoints;
}

vector<cv::Point2f> ofxFaceTracker2Landmarks::getCvImagePoints() const {
    int n = shape.num_parts();
    vector<cv::Point2f> imagePoints(n);
    for(int i = 0; i < n; i++) {
        imagePoints[i] = ofxCv::toCv(getImagePoint(i));
    }
    return imagePoints;
}



ofPolyline ofxFaceTracker2Landmarks::getImageFeature(Feature feature) const {
    return getFeature(feature, getImagePoints());
}

ofMesh ofxFaceTracker2Landmarks::getImageMesh() const{
    return getMesh(getCvImagePoints());
}


vector<int> ofxFaceTracker2Landmarks::getFeatureIndices(Feature feature) {
    switch(feature) {
        case LEFT_EYE_TOP: return consecutive(36, 40);
        case RIGHT_EYE_TOP: return consecutive(42, 46);
        case LEFT_JAW: return consecutive(0, 9);
        case RIGHT_JAW: return consecutive(8, 17);
        case JAW: return consecutive(0, 17);
        case LEFT_EYEBROW: return consecutive(17, 22);
        case RIGHT_EYEBROW: return consecutive(22, 27);
        case LEFT_EYE: return consecutive(36, 42);
        case RIGHT_EYE: return consecutive(42, 48);
        case OUTER_MOUTH: return consecutive(48, 60);
        case INNER_MOUTH: return consecutive(60, 68);
        case NOSE_BRIDGE: return consecutive(27, 31);
        case NOSE_BASE: return consecutive(31, 36);
        case FACE_OUTLINE: {
            static int faceOutline[] = {17,18,19,20,21,22,23,24,25,26, 16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};
            return vector<int>(faceOutline, faceOutline + 27);
        }
        case ALL_FEATURES: return consecutive(0, 68);
    }
}


template <class T>
ofPolyline ofxFaceTracker2Landmarks::getFeature(Feature feature, vector<T> points) const {
    ofPolyline polyline;
    vector<int> indices = getFeatureIndices(feature);
    for(int i = 0; i < indices.size(); i++) {
        int cur = indices[i];
        polyline.addVertex(points[cur]);
    }
    switch(feature) {
        case LEFT_EYE:
        case RIGHT_EYE:
        case OUTER_MOUTH:
        case INNER_MOUTH:
        case FACE_OUTLINE:
            polyline.close();
            break;
        default:
            break;
    }
    
    return polyline;
}

vector<int> ofxFaceTracker2Landmarks::consecutive(int start, int end) {
    int n = end - start;
    vector<int> result(n);
    for(int i = 0; i < n; i++) {
        result[i] = start + i;
    }
    return result;
}

template <class T>
ofMesh ofxFaceTracker2Landmarks::getMesh(vector<T> points) const {
    cv::Rect rect(0, 0, info.inputWidth, info.inputHeight);
    cv::Subdiv2D subdiv(rect);
    
	map<int,int> sdPts;
	for(size_t i=0;i<points.size();i++) {
        if( rect.contains(points[i])) {
			int id = subdiv.insert(points[i]);
			sdPts[id] = i;
		}
    }
    
	// delaunay
    vector<cv::Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    
    ofMesh mesh;
    mesh.setMode(OF_PRIMITIVE_TRIANGLES);

	// add the mesh vertices, which we later index per delaunay results
	mesh.addVertices(ofxCv::toOf(points).getVertices());

	// add texture coords in pixel dims or normalized
	ofVec2f div =  ofGetUsingArbTex() ? ofVec2f(1.) : ofVec2f(rect.width, rect.height);
	for (auto& vt : mesh.getVertices()) {
		mesh.addTexCoord(vt / div);
	}

	
	for (size_t i = 0; i < triangleList.size(); i++)
	{
		cv::Vec6f& t = triangleList[i];

		// find indices in mesh vertices corresponding to delaunay result
		vector<ofIndexType> indices;
		for (int j = 0; j < 3; j++) {
			cv::Point2f pos = cv::Point2f(t[j * 2], t[j * 2 + 1]);
			if (rect.contains(pos)) {
				int sdId = subdiv.findNearest(pos);
				int ptId = -1;
				try {
					ptId = sdPts.at(sdId);
				}
				catch (...){
					ofLogError("ofxFaceTracker2Landmarks") << "can't find subdiv point [" << sdId << "] in image mesh verts...";
				}
				if (ptId >= 0 && ptId < mesh.getNumVertices()) {
					indices.push_back(ptId);
				}
			}
		}
		if (indices.size() == 3) {
			// add mesh indices if we've found a whole triangle
			mesh.addIndices(indices);
		}
        
    }
    return mesh;
    
}

