Environment Setup, Data Preprocessing and Code Execution

1. Download and install Ananconda (https://www.continuum.io/downloads)

2. Download and install GPU libraries (CUDA, nvidia-smi: https://developer.nvidia.com/cuda-downloads) 

3. Install Keras for GPU use after installing the required dependencies (https://keras.io/#installation)

4. Install Tensor Flow for GPU (https://www.tensorflow.org/install/install_linux#NVIDIARequirements)

5. Modify the keras.json  file according to the GPU backend. Keras.json can be found in ~/.keras/keras.json
 	
	Specify ‘tensorflow’/’theano’ in ‘backend’ filed accordingly.
	
	The file looks like this:
	{
   		 "epsilon": 1e-07,
   	 	"floatx": "float32",
    		"image_data_format": "channels_last",
   	 	"backend": "tensorflow"
	}

6. Download raw data (http://www.openu.ac.il/home/hassner/Adience/data.html#agegender)

7. Run the data pre processing script from the code folder. 

	a. Clone the github repository: https://github.com/GilLevi/AgeGenderDeepLearning

	b. Specify the correct path for cloned folder in “images_path_file” in convertImages_gender.py/convertImages.age.py

	c. Specify path for actual dataset images “aligned” folder in variable “actual_images” 

	d. Run convertImages_gender.py /  convertImages_age.py

8. Set up proper paths and run config in constants.py

9. Run the runproject.py script to execute code with options

10. Run individual script file to run specific code.

	Option 1: Run train_base_gender.py  for training base Gender Model
	Option 2: Run train_base_age.py for training base Age model
	Option 3: Run  train_gender_seeded_by_age.py for training Age model and retrain Gender model with initialized weights
	Option 4: Run  train_age_seeded_by_gender.py for training Gender model and retrain Age model with initialized weights
	Option 5: Run  train_gender_seeded_by_age_FREEZED.py for training Age model and retrain Gender model using Transfer Learning
	Option 6: Run  train_age_seeded_by_gender_FREEZED.py for training Gender model and retrain Age model using Transfer Learning
	Option 7: Run  train_gender_seeded_by_age_DENSE_UNFROZEN.py for training Age model and retrain Gender model using Transfer Learning with dense layer unfrozen
	Option 8: Run  train_age_seeded_by_gender_DENSE_UNFROZEN.py for training Gender model and retrain Age model using Transfer Learning with dense layer unfrozen

