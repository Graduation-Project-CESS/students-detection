Note that":
"my_image.jpg" is a temporary image, which is used to store the images in the code & locally in "loaded_images" directory

Dependencies:
Install visual studio (2019 recomended).
Install CMake from this link https://github.com/Kitware/CMake/releases/download/v3.19.0-rc2/cmake-3.19.0-rc2-win64-x64.msi

create a new environment called face detection from anaconda prompt
	conda create --name face-detection python=3.7
	source activate face-detection
	
Install some packages from anaconda prompt
	conda install scikit-learn
	conda install -c conda-forge scikit-image
	conda install -c menpo opencv3
	pip install cvlib from anaconda prompt #reference https://github.com/arunponnusamy/cvlib
	pip install dlib
	pip install face_recognition #reference https://pypi.org/project/face-recognition/

If you faced any issues while installing dlib or face_recognition please follow this link https://github.com/ageitgey/face_recognition/issues/175

install opencv, tensorflow packages from anaconda navigator
