#include "CodingEngine.cuh"


/*
* Memory initialization function which preallocates every memory needed in the process.
* It takes in consideration everything from Image/Video, Lossy/Lossless and GrayScale/RGB.
*/
void CodingEngine::initMemory(bool typeOfCoding)
{
	SupportFunctions::fixImageProportions(this->_frameStructure, CBLOCK_LENGTH, CBLOCK_WIDTH);
	if (typeOfCoding == IMAGE)
	{
		int DDataExtra = 0;
		int DDataExtra2 = 0;
		int size = _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(int);

		if (_frameStructure->getIsRGB() == true)
		{
			int sizeOfImage = _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight();
			_HImagePixelsCharRGB = new unsigned char*[3];
			_DImagePixelsCharRGB = new unsigned char*[3];
			cudaHostAlloc(&(_HImagePixelsCharRGB[0]), _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight(), 0);
			cudaHostAlloc(&(_HImagePixelsCharRGB[1]), _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight(), 0);
			cudaHostAlloc(&(_HImagePixelsCharRGB[2]), _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight(), 0);
			GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsCharRGB[0], _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight()));
			GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsCharRGB[1], _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight()));
			GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsCharRGB[2], _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight()));

			if (_waveletType == LOSSLESS)
			{
				_DImagePixelsRGBTransformed = new int*[3];
				for (int i = 1; i<_waveletLevels; ++i)
					DDataExtra2 += (_frameStructure->getAdaptedWidth() / (2 << (i - 1)))* (_frameStructure->getAdaptedHeight() / (2 << (i - 1)));
				
				GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsRGBTransformed[0], size + DDataExtra2 * sizeof(int)));
				GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsRGBTransformed[1], size + DDataExtra2 * sizeof(int)));
				GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsRGBTransformed[2], size + DDataExtra2 * sizeof(int)));
			}
			else
			{
				_DImagePixelsRGBTransformedLossy = new float*[3];
				GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsRGBTransformedLossy[0], _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(float)));
				GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsRGBTransformedLossy[1], _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(float)));
				GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsRGBTransformedLossy[2], _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(float)));
				
			}
		}
		else
		{

			//Memory Allocation for the DWT Coding Process.
			GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsChar, _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight()));
			if (_frameStructure->getSignedOrUnsigned() == 0)
			{
				if (_waveletType == LOSSY)
				{
					GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsLossy, _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(float)));
				}
				else
					GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixels, _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(int)));
					
			}
		}

		for (int i = 1; i<_waveletLevels; ++i)
			DDataExtra += (_frameStructure->getAdaptedWidth() / (2 << (i - 1)))* (_frameStructure->getAdaptedHeight() / (2 << (i - 1)));
		if (_waveletType == LOSSLESS)
		{
			GPU_HANDLE_ERROR(cudaMalloc(&_DWaveletCoefficients, size + DDataExtra * sizeof(int)));
		}
		else
			GPU_HANDLE_ERROR(cudaMalloc(&_DWaveletCoefficientsLossy, size + DDataExtra * sizeof(float)));

		//Memory Allocation for the BPC Coding Process.
		GPU_HANDLE_ERROR(cudaMalloc(&_DCodeStreamValues, size));

		GPU_HANDLE_ERROR(cudaMalloc(&_DSizeArray, (int)ceil(_frameStructure->getAdaptedWidth() / (float)CBLOCK_WIDTH) * (int)ceil(_frameStructure->getAdaptedHeight() / (float)CBLOCK_LENGTH) * sizeof(int)));
		GPU_HANDLE_ERROR(cudaMalloc(&_DPrefixedArray, ((int)ceil(_frameStructure->getAdaptedWidth() / (float)CBLOCK_WIDTH) * (int)ceil(_frameStructure->getAdaptedHeight() / (float)CBLOCK_LENGTH)) * sizeof(int) + sizeof(int)));
		
		int storage = _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight() / (CBLOCK_WIDTH*CBLOCK_LENGTH * 2);
		//This way we make sure that for really small frames or images, the temporal storage needed by CUB is covered. For really big images or frames, the equation above is enough - tested empirically.
		if (storage < 1000)
			storage = 1000;
		GPU_HANDLE_ERROR(cudaMalloc(&_DTempStoragePArray, storage));

		_HLUTBSTableSteps = 256;
		_HTotalBSSize = (int*)malloc(sizeof(int));
		GPU_HANDLE_ERROR(cudaMalloc(&_DLUTBSTable, _HLUTBSTableSteps * sizeof(int) + 4));

		//Memory Allocation for BitCon Coding Process
		_HExtraInformation = (unsigned short*)malloc(9 * sizeof(unsigned short));
		GPU_HANDLE_ERROR(cudaMalloc(&_DBitStreamValues, _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(unsigned short)));
		_HBitStreamValues = (unsigned short*)malloc(_frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(unsigned short));
	}
}

void CodingEngine::readGrayScaleImage()
{
	SupportFunctions::markInitProfilerCPUSection("IO", "Disk Reading");
	IOManager<int, int2> *IOM = new IOManager<int, int2>();
	int sizeOfImage = _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight();
	cudaHostAlloc(&_HImagePixelsChar, _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight(), 0);
	IOM->loadFrameCAdaptedSizes(_frameStructure, _HImagePixelsChar, 0);
	GPU_HANDLE_ERROR(cudaMemcpyAsync(_DImagePixelsChar, _HImagePixelsChar, _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(unsigned char), cudaMemcpyHostToDevice, cudaStreamDefault));
	delete IOM;
	SupportFunctions::markEndProfilerCPUSection();
}

void CodingEngine::readRGBImage()
{
	SupportFunctions::markInitProfilerCPUSection("IO", "Disk Reading");
	IOManager<int, int2> *IOM = new IOManager<int, int2>();

	for (int i = 0; i < 3; i++)
	{
		 auto loadFrames_s = std::chrono::steady_clock::now();
		 IOM->loadFrameCAdaptedSizes(_frameStructure, _HImagePixelsCharRGB[i], i);
		 auto loadFrames_f = std::chrono::steady_clock::now();
		 double RGBRead = std::chrono::duration_cast<std::chrono::duration<double>>(loadFrames_f - loadFrames_s).count();
		 std::cout << "RGB Read acum time is " << i << ": " << RGBRead << std::endl;
		 
		 GPU_HANDLE_ERROR(cudaMemcpyAsync(_DImagePixelsCharRGB[i], _HImagePixelsCharRGB[i], _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(unsigned char), cudaMemcpyHostToDevice, cudaStreamDefault));
	}
	delete IOM;
	SupportFunctions::markEndProfilerCPUSection();
}

/*
* Kernel which launches the lossless color transformation, changing from RGB color space to YCbCr color space. It also reduces the size of the samples by applying an offset if the data type is unsigned.
*/
__global__ void RGBTransformLossless(unsigned char* inputR, unsigned char* inputG, unsigned char* inputB, int* outputR, int* outputG, int* outputB, int bitdepth, bool uSigned)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	outputR[threadId] = (float)inputR[threadId];
	outputG[threadId] = (float)inputG[threadId];
	outputB[threadId] = (float)inputB[threadId];

}

/*
* Kernel which launches the lossy color transformation, changing from RGB color space to YCbCr color space. It also reduces the size of the samples by applying an offset if the data type is unsigned.
*/
__global__ void RGBTransformLossy(unsigned char* inputR, unsigned char* inputG, unsigned char* inputB, float* outputR, float* outputG, float* outputB, int bitdepth, bool uSigned)
{
	int offset = 0;
	if (uSigned == false)
	{
		offset = 1 << (bitdepth - 1);
	}
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	float componentR = (float)inputR[threadId] - offset;
	float componentG = (float)inputG[threadId] - offset;
	float componentB = (float)inputB[threadId] - offset;

	float componentRTransformed = _irreversibleColorTransformForward[0][0] * componentR + _irreversibleColorTransformForward[0][1] * componentG + _irreversibleColorTransformForward[0][2] * componentB;
	float componentGTransformed = _irreversibleColorTransformForward[1][0] * componentR + _irreversibleColorTransformForward[1][1] * componentG + _irreversibleColorTransformForward[1][2] * componentB;
	float componentBTransformed = _irreversibleColorTransformForward[2][0] * componentR + _irreversibleColorTransformForward[2][1] * componentG + _irreversibleColorTransformForward[2][2] * componentB;

	outputR[threadId] = componentRTransformed;
	outputG[threadId] = componentGTransformed;
	outputB[threadId] = componentBTransformed;
}

/*
* Host function which launches the color transformation for images
*/
void CodingEngine::prepareRGBImage(cudaStream_t mainStream)
{
	if (_waveletType == LOSSLESS)
	{
		int numberOfThreadsPerBlock = 256;
		int numberOfBlocks = (int)ceil(_frameStructure->getAdaptedHeight() * _frameStructure->getAdaptedWidth() / numberOfThreadsPerBlock);
		
		auto strartThreads = std::chrono::steady_clock::now();
		RGBTransformLossless <<<numberOfBlocks, numberOfThreadsPerBlock, 0, mainStream>>>	(_DImagePixelsCharRGB[0], _DImagePixelsCharRGB[1], _DImagePixelsCharRGB[2], _DImagePixelsRGBTransformed[0], _DImagePixelsRGBTransformed[1], _DImagePixelsRGBTransformed[2], _frameStructure->getBitDepth(), _frameStructure->getSignedOrUnsigned());
		auto finishThreads= std::chrono::steady_clock::now();
		double Threads_time = std::chrono::duration_cast<std::chrono::duration<double>>(finishThreads - strartThreads).count();
		std::cout << "RGBTransformLossless<<: " << Threads_time << std::endl;

		auto strartStream = std::chrono::steady_clock::now();
		cudaStreamSynchronize(mainStream);
		auto finishStream = std::chrono::steady_clock::now();
		double Stream = std::chrono::duration_cast<std::chrono::duration<double>>(finishStream - strartStream).count();
		std::cout << "cudaStreamSynchronize: " << Stream << std::endl;

		auto strartError = std::chrono::steady_clock::now();
		KERNEL_ERROR_HANDLER;
		auto finishError= std::chrono::steady_clock::now();
		double Error = std::chrono::duration_cast<std::chrono::duration<double>>(finishError - strartError).count();
		std::cout << "KERNEL_ERROR_HANDLER: " << Error << std::endl;
	}
	else
	{
		int numberOfThreadsPerBlock = 256;
		int numberOfBlocks = (int)ceil(_frameStructure->getAdaptedHeight() * _frameStructure->getAdaptedWidth() / numberOfThreadsPerBlock);
		RGBTransformLossy << <numberOfBlocks, numberOfThreadsPerBlock, 0, mainStream >> >	(_DImagePixelsCharRGB[0], _DImagePixelsCharRGB[1], _DImagePixelsCharRGB[2], _DImagePixelsRGBTransformedLossy[0], _DImagePixelsRGBTransformedLossy[1], _DImagePixelsRGBTransformedLossy[2], _frameStructure->getBitDepth(), _frameStructure->getSignedOrUnsigned());
		cudaStreamSynchronize(mainStream);
		KERNEL_ERROR_HANDLER;
	}
}

/*
* Function which calls the IO Manager to write a grayscale image.
*/
void CodingEngine::writeCodedBitStream()
{
	IOManager<unsigned short, ushort2> *IOM = new IOManager<unsigned short, ushort2>();
	IOM->writeBitStreamFile(_HBitStreamValues, _HTotalBSSize[0], _outputFile);
	delete IOM;
}

/*
* Function which calls the IO Manager to write an RGB image.
*/
void CodingEngine::writeRGBBitStream(cudaStream_t mainStream, int iter)
{
	IOManager<unsigned short, ushort2> *IOM = new IOManager<unsigned short, ushort2>();
	if (iter == 0)
	{
		IOM->replaceExistingFile(_outputFile);
		IOM->replaceExistingFile(_outputFile + "_SIZE");
	}
	IOM->writeCodedFrame(_frameStructure, _HBitStreamValues, iter, _HTotalBSSize[0], _outputFile);
	delete IOM;
}

/*
* Transforms the image taken into the proper data type.
*/
template<class T>
__global__ void offsetImage(unsigned char* inputData, T* outputData, int bitDepth)
{
	int threadId = threadIdx.x + blockIdx.x*blockDim.x;
	T charData = (T)(inputData[threadId]);
	charData = charData - (1 << (bitDepth - 1));
	outputData[threadId] = charData;
}

/*
* Function which manages the general flow of instructions to code images, either in lossy / lossless or grayscale / RGB.
*/
void CodingEngine::runImage()
{
	if (_waveletType == LOSSLESS)
	{

		if (_frameStructure->getIsRGB())
		{
			auto readRGB_start = std::chrono::steady_clock::now();
			readRGBImage();
			auto readRGB_finish = std::chrono::steady_clock::now();
			double RGBRead = std::chrono::duration_cast<std::chrono::duration<double>>(readRGB_finish - readRGB_start).count();
			std::cout << "RGB Read acum time is: " << RGBRead << std::endl;


			auto startProcessDisregardAllocationTimings = std::chrono::steady_clock::now();
			SupportFunctions::markInitProfilerCPUSection("ColorTransform", "Color Tranform Kernel");
			prepareRGBImage(cudaStreamDefault);
			auto prepare_finish = std::chrono::steady_clock::now();
			double RGBinit = std::chrono::duration_cast<std::chrono::duration<double>>(prepare_finish - startProcessDisregardAllocationTimings).count();
			std::cout << "RGB Init acum time is: " << RGBinit << std::endl;
			SupportFunctions::markEndProfilerCPUSection();
			for (int i = 0; i < 3; i++)
			{
				SupportFunctions::markInitProfilerCPUSection("BPC", "BPC");
				BPCCuda<int>* BPC = new BPCCuda<int>(_frameStructure, _DImagePixelsRGBTransformed[i], _waveletLevels, _DWTCBWidth, _DWTCBHeight, _codingPasses, _waveletType, _quantizationSize, _k, _LUTAmountOfBitplaneFiles);
				BPC->Code(_LUTNumberOfBitplanes, _LUTNumberOfSubbands, _LUTContextRefinement, _LUTContextSign, _LUTContextSignificance, _LUTMultPrecision, _LUTInformation[i], _DCodeStreamValues, _DPrefixedArray, _DTempStoragePArray, _DSizeArray, _HExtraInformation, _DBitStreamValues, _HTotalBSSize, _DLUTBSTable, _HLUTBSTableSteps, i, cudaStreamDefault, _numberOfFrames, &_measurementsBPC[0]);
				SupportFunctions::markEndProfilerCPUSection();
				SupportFunctions::markInitProfilerCPUSection("Copying", "Writing to disk");

				GPU_HANDLE_ERROR(cudaMemcpy(_HBitStreamValues, _DBitStreamValues, (_HTotalBSSize[0] * sizeof(unsigned short)), cudaMemcpyDeviceToHost));
				this->writeRGBBitStream(cudaStreamDefault, i);

				delete BPC;
				SupportFunctions::markEndProfilerCPUSection();
			}
			auto finishProcessDisregardAllocationTimings = std::chrono::steady_clock::now();
			double elapsedTimeProcessDisregardAllocationTimings = std::chrono::duration_cast<std::chrono::duration<double>>(finishProcessDisregardAllocationTimings - startProcessDisregardAllocationTimings).count();
			std::cout << "The time spent with the app without considering allocation periods is: " << elapsedTimeProcessDisregardAllocationTimings << std::endl;
			std::cout << "RGB Init 2 acum time is: " << RGBinit << std::endl;
			std::cout << "DWT Init acum time is: " << *_measurementsDWTInit << std::endl;
			std::cout << "DWT Kernel acum time is: " << _measurementsDWT[0] << std::endl;
			std::cout << "BPC Kernel acum time is: " << _measurementsBPC[0] << std::endl;
		}
	}
}

/*
* General function which initializes the CPU threads needed depending on the amount of GPU streams used.
*/
void CodingEngine::engineManager(int cType)
{
	if (cType == IMAGE)
	{
		_measurementsRGB = new double;
		_measurementsDWTInit = new double;
		_measurementsBPC = new double[1];
		_measurementsDWT = new double[1];
		initLUT();
		auto start1 = std::chrono::steady_clock::now();
		initMemory(IMAGE);
		auto finish1 = std::chrono::steady_clock::now();
		double elapsed_seconds1 = std::chrono::duration_cast<std::chrono::duration<double>>(finish1 - start1).count();
		std::cout << "The time spent with the initMemory is: " << elapsed_seconds1 << std::endl;
		runImage();
	}
}


/*
* Functions to try other functionalities. Are not used
*/


/*
* Host function which launches the color transformation for images
*/
void CodingEngine::prepareGrayScaleImage(cudaStream_t mainStream)
{
	int numberOfThreadsPerBlock = 256;
	int numberOfBlocks = (int)ceil(_frameStructure->getAdaptedHeight() * _frameStructure->getAdaptedWidth() / numberOfThreadsPerBlock);
	
	expandShortToIntKernel<<<numberOfBlocks, numberOfThreadsPerBlock, 0, mainStream>>>((unsigned short*)_DImagePixelsShort, _DImagePixels, _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight());
	cudaStreamSynchronize(mainStream);
	KERNEL_ERROR_HANDLER;
}



/*
* Kernel which launches the lossless grayscale transformation
*/
__global__ void expandShortToIntKernel(unsigned short* input, int* output, int size)
{

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadId < size) 
		{
			output[threadId] = (int)input[threadId];
		}
}

/*
* Bridge
*/
void CodingEngine::Bridge(cudaStream_t mainStream, int* input, int* output)
{
	int numberOfThreadsPerBlock = 256;
	int numberOfBlocks = (int)ceil(_frameStructure->getAdaptedHeight() * _frameStructure->getAdaptedWidth() / numberOfThreadsPerBlock);
	
	KernelBridge<<<numberOfBlocks, numberOfThreadsPerBlock, 0, mainStream>>>((unsigned int*)input, output, _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight());
	cudaStreamSynchronize(mainStream);
	KERNEL_ERROR_HANDLER;
}

/*
* Kernel which launches the lossless grayscale transformation
*/
__global__ void KernelBridge(unsigned int* input, int* output, int size)
{

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadId < size) 
		{
			output[threadId] = (int)input[threadId];
		}
}

__global__ void RGBTransformLossless2(unsigned char* inputR, unsigned char* inputG, unsigned char* inputB, int* outputR, int* outputG, int* outputB, int bitdepth, bool uSigned, int height, int width)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;

    if (threadId >= totalPixels) return;
    int halfIndex = threadId / 2;
    int halfOffset = (totalPixels + 1) / 2;

    if (threadId % 2 == 0) {
        outputR[halfIndex] = (int)inputR[threadId];
        outputR[halfIndex + halfOffset] = (int)inputG[threadId];
        outputB[threadId] = (int)inputB[threadId];
    }
    else {
        
        outputG[halfIndex] = (int)inputR[threadId];
        outputG[halfIndex + halfOffset] = (int)inputG[threadId];
        outputB[threadId] = (int)inputB[threadId];
    }
}