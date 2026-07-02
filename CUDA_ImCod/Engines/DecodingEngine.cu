#include "DecodingEngine.cuh"

/*
* Memory initialization function which preallocates every memory needed in the process.
* It takes in consideration everything from Image/Video, Lossy/Lossless and GrayScale/RGB.
*/
void DecodingEngine::initMemory(bool typeOfCoding)
{
	if (typeOfCoding == IMAGE)
	{
		_extraWaveletAllocation = 0;
		_HBasicInformation = (int*)malloc(16 * sizeof(int));

		retrieveBasicImageInformation(_frameStructure->getName());
		_frameStructure->setWidth(_HBasicInformation[0] / _HBasicInformation[10]);
		_frameStructure->setHeight(_HBasicInformation[10]);
		_frameStructure->setBitDepth(_HBasicInformation[5]);
		_frameStructure->setComponents(_HBasicInformation[8]);
		_frameStructure->setBitsPerSample(_HBasicInformation[12]);
		_frameStructure->setEndianess(_HBasicInformation[11]);
		_frameStructure->setIsRGB(_HBasicInformation[9]);
		_frameStructure->setSignedOrUnsigned(_HBasicInformation[13]);
		SupportFunctions::fixImageProportions(this->_frameStructure, CBLOCK_LENGTH, CBLOCK_WIDTH);
		_frameSize = _frameStructure->getAdaptedHeight() * _frameStructure->getAdaptedWidth();
		this->setCodingPasses(_HBasicInformation[1]);
		this->setCBHeight(_HBasicInformation[2]);
		this->setCBWidth(_HBasicInformation[3]);
		this->setWaveletLevels(_HBasicInformation[4]);
		this->setWType(_HBasicInformation[6]);
		this->setQSizes(_HBasicInformation[7] / 10000.0);
		this->setNumberOfFrames(_HBasicInformation[14]);
		this->setKFactor(_HBasicInformation[15]/1000.0);
 		_HLUTBSTableSteps = 256;

		for (int i = 1; i<_HBasicInformation[4]; ++i)
			_extraWaveletAllocation += _frameSize / (pow(4, i));

		if (_frameStructure->getIsRGB() == true)
		{
			int sizeOfImage = _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight();
			_bufferReadingValueRGB = new int[3];
			_bufferReadingValueRGB[0] = 0;
			_bufferReadingValueRGB[1] = 0;
			_bufferReadingValueRGB[2] = 0;
			_HBitStreamValuesRGB = new unsigned short*[3];
			_HImagePixelsCharRGB = new unsigned char*[3];
			_DImagePixelsCharRGB = new unsigned char*[3];
			cudaHostAlloc(&(_HBitStreamValuesRGB[0]), _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight() * sizeof(unsigned short), 0);
			cudaHostAlloc(&(_HBitStreamValuesRGB[1]), _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight() * sizeof(unsigned short), 0);
			cudaHostAlloc(&(_HBitStreamValuesRGB[2]), _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight() * sizeof(unsigned short), 0);
			cudaHostAlloc(&(_HImagePixelsCharRGB[0]), _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight(), 0);
			cudaHostAlloc(&(_HImagePixelsCharRGB[1]), _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight(), 0);
			cudaHostAlloc(&(_HImagePixelsCharRGB[2]), _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight(), 0);
			GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsCharRGB[0], _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight()));
			GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsCharRGB[1], _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight()));
			GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsCharRGB[2], _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight()));

			if (_waveletType == LOSSLESS)
			{
				_DImagePixelsRGBTransformed = new int*[3];
				GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsRGBTransformed[0], (_frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() + _extraWaveletAllocation) * sizeof(int)));
				GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsRGBTransformed[1], (_frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() + _extraWaveletAllocation) * sizeof(int)));
				GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsRGBTransformed[2], (_frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() + _extraWaveletAllocation) * sizeof(int)));

			}
			else
			{
				_DImagePixelsRGBTransformedLossy = new float*[3];
				GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsRGBTransformedLossy[0], (_frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() + _extraWaveletAllocation) * sizeof(float)));
				GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsRGBTransformedLossy[1], (_frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() + _extraWaveletAllocation) * sizeof(float)));
				GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsRGBTransformedLossy[2], (_frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() + _extraWaveletAllocation) * sizeof(float)));

			}
		}
		else
		{
			cudaHostAlloc(&_HBitStreamValues, _frameSize * sizeof(unsigned short), 0);
			if (_waveletType == LOSSLESS)
			{
				cudaHostAlloc(&_HImagePixels, _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(int), 0);
				GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixels, (_frameSize + _extraWaveletAllocation) * sizeof(int)));
			}
			else
			{
				cudaHostAlloc(&_HImagePixelsLossy, _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(float), 0);
				GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsLossy, (_frameSize + _extraWaveletAllocation) * sizeof(float)));
			}
		}

		int storage = _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight() / (CBLOCK_WIDTH*CBLOCK_LENGTH * 2);
		//This way we make sure that for really small frames or images, the temporal storage needed by CUB is covered. For really big images or frames, the equation above is enough - tested empirically.
		if (storage < 1000)
			storage = 1000;

		GPU_HANDLE_ERROR(cudaMalloc(&_DBitStreamValues, _frameSize * sizeof(unsigned short)));
		GPU_HANDLE_ERROR(cudaMalloc(&_DSizeArray, (int)ceil(_frameSize / ((float)CBLOCK_WIDTH * (float)CBLOCK_LENGTH)) * sizeof(int)));
		GPU_HANDLE_ERROR(cudaMalloc(&_DPrefixedArray, (int)ceil(_frameSize / ((float)CBLOCK_WIDTH * (float)CBLOCK_LENGTH)) * sizeof(int)));
		GPU_HANDLE_ERROR(cudaMalloc(&_DWaveletCoefficients, _frameSize * sizeof(int)));
		GPU_HANDLE_ERROR(cudaMalloc(&_DCodeStreamValues, _frameSize * sizeof(int)));
		GPU_HANDLE_ERROR(cudaMalloc(&_DLUTBSTable, (_HLUTBSTableSteps + 1) * sizeof(int)));
		GPU_HANDLE_ERROR(cudaMalloc(&_DTempStoragePArray, storage));
		_HSizeArray = (int*)malloc((_frameSize / (CBLOCK_WIDTH * CBLOCK_LENGTH)) * sizeof(int));
		_HTotalBSSize = (int*)malloc(sizeof(int));
	}
}

bool DecodingEngine::readRGBCompressedBitStream()
{
	bool ret = false;
	SupportFunctions::markInitProfilerCPUSection("IO", "Disk Reading");
	IOManager<unsigned short, ushort2>* IOM = new IOManager<unsigned short, ushort2>(_frameStructure->getName());
	cudaHostAlloc(&_componentSizes, 3 * sizeof(int), 0);
	IOM->readBulkSizes(_componentSizes, _frameStructure, 3);
	long long int offset = 0;
	int i = 0;
	for (i; i < 3; i++)
	{
		offset = offset + IOM->loadCodedFrame(_frameStructure, _HBitStreamValuesRGB[i], i, _componentSizes[i], offset);
		_bufferReadingValueRGB[i] = 1;
	}
	//We coded the bitStream file into a variable. Now, decode time.
	if (i == 3)
		ret = true;
	SupportFunctions::markEndProfilerCPUSection();
	return ret;
}

void DecodingEngine::writeDecompressedImage()
{
	if (_waveletType == LOSSLESS)
	{
		IOManager<int, int2>* IOM = new IOManager<int, int2>(_frameStructure->getName());
		IOM->writeImage(_HImagePixels, _frameStructure->getWidth(), _frameStructure->getHeight(), _frameStructure->getBitDepth(), _outputFile);
	}
	else
	{
		IOManager<float, float2>* IOM = new IOManager<float, float2>(_frameStructure->getName());
		IOM->writeImage(_HImagePixelsLossy, _frameStructure->getWidth(), _frameStructure->getHeight(), _frameStructure->getBitDepth(), _outputFile);
	}
}

void DecodingEngine::writeRGBImage()
{
	IOManager<int, int2> *IOM = new IOManager<int, int2>();
	IOM->replaceExistingFile(_outputFile);
	SupportFunctions::markInitProfilerCPUSection("writeCodedFile", "writeCodedFile");
	IOM->writeDecodedFrameUChar(_frameStructure, _HImagePixelsCharRGB, _outputFile);
	SupportFunctions::markEndProfilerCPUSection();

}

/*
* Retrieves the information from a coded video / image and extracts the side information needed to decode it.
*/
void DecodingEngine::getExtraInformation()
{
	_HBasicInformation[0] = (_HExtraInformation[0] | (_HExtraInformation[1] << 16)); //Image Size
	_HBasicInformation[1] = (_HExtraInformation[2] & 1) == 1 ? 3 : 2; // Coding Passes
	_HBasicInformation[2] = ((_HExtraInformation[2] >> 1) & ((1 << 7) - 1)); // DWT Height
	_HBasicInformation[3] = ((_HExtraInformation[2] >> 8) & ((1 << 7) - 1)); // DWT Width
	_HBasicInformation[4] = (((_HExtraInformation[2] >> 15 & 1)) | (_HExtraInformation[3] & 7) << 1); // WLevels
	_HBasicInformation[5] = (_HExtraInformation[3] >> 3) & ((1 << 7) - 1); // BitDepth
	_HBasicInformation[6] = (_HExtraInformation[3] >> 10) & 1; // WType
	_HBasicInformation[7] = (((_HExtraInformation[3] >> 11) & 31) | (_HExtraInformation[4] & 511) << 5); // QSize
	_HBasicInformation[8] = ((_HExtraInformation[4] >> 9) & 127) | ((_HExtraInformation[5] & 127) << 9); // Components
	_HBasicInformation[9] = (_HExtraInformation[5] >> 7) & 1; // RGB
	_HBasicInformation[10] = ((_HExtraInformation[5] >> 8) & 255) | ((_HExtraInformation[6] & 255) << 8); // Image Height (used to recover width as well from the total size)
	_HBasicInformation[11] = ((_HExtraInformation[6] >> 8) & 1); // Endianess
	_HBasicInformation[12] = (_HExtraInformation[6] >> 9) & ((1<<5)-1); // BPS
	_HBasicInformation[13] = (_HExtraInformation[6] >> 14) & 1; // Signed/Unsigned
	_HBasicInformation[14] = (((_HExtraInformation[6] >> 15) & 1) | _HExtraInformation[7] << 1); // Number of frames
	_HBasicInformation[15] = _HExtraInformation[8]; // KFactor value
}

void DecodingEngine::retrieveBasicImageInformation(std::string inputFile)
{
	IOManager<unsigned short, ushort2> *IOM = new IOManager<unsigned short, ushort2>();
	_HExtraInformation = (unsigned short*)malloc(9 * sizeof(unsigned short));
	IOM->loadBasicInfo(_HExtraInformation, 9, inputFile);
	getExtraInformation();
	delete IOM;
}

/*
* Kernel which launches the reverse lossless color transformation, changing from YCbCr color space to RGB color space. It also recovers the size of the samples by applying an offset if the data type is unsigned.
*/ 
__global__ void RGBTransformLossless(int* inputR, int* inputG, int* inputB, unsigned char* outputR, unsigned char* outputG, unsigned char* outputB, int bitdepth, bool uSigned)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	outputR[threadId] = (float)inputR[threadId];
	outputG[threadId] = (float)inputG[threadId];
	outputB[threadId] = (float)inputB[threadId];
}


/*
* Host function which launches the color transformation for images
*/
void DecodingEngine::prepareRGBImage(cudaStream_t mainStream)
{

	if (_waveletType == LOSSLESS)
	{
		int numberOfThreadsPerBlock = 256;
		int numberOfBlocks = (int)ceil(_frameStructure->getHeight() * _frameStructure->getWidth() / numberOfThreadsPerBlock);
		RGBTransformLossless << <numberOfBlocks, numberOfThreadsPerBlock, 0, mainStream >> >	(_DImagePixelsRGBTransformed[0] + _extraWaveletAllocation, _DImagePixelsRGBTransformed[1] + _extraWaveletAllocation, _DImagePixelsRGBTransformed[2] + _extraWaveletAllocation, _DImagePixelsCharRGB[0], _DImagePixelsCharRGB[1], _DImagePixelsCharRGB[2], _frameStructure->getBitDepth(), _frameStructure->getSignedOrUnsigned());
		cudaStreamSynchronize(mainStream);
		KERNEL_ERROR_HANDLER;
		for (int i = 0; i < 3; i++)
			GPU_HANDLE_ERROR(cudaMemcpy(_HImagePixelsCharRGB[i], _DImagePixelsCharRGB[i], _frameStructure->getWidth() * _frameStructure->getHeight() * sizeof(char), cudaMemcpyDeviceToHost));
	}
}

/*
* Transforms the image taken into the proper data type.
*/
__global__ void removeOffsetAndApplyMaxMinLossy(float *data, int bitDepth, int signedOrUnsigned)
{
	int offset = 0;
	if (signedOrUnsigned == 0)
		offset = 1 << (bitDepth - 1);

	int threadId = threadIdx.x + blockIdx.x * blockDim.x;
	//0.01f used to avoid cases in which CUDA is not rounding correctly x.5 values. For example, 4.5 is rounded to 4 instead of 5, but 4.51 is correctly rounded to 5.
	data[threadId] = fmaxf(fminf(__float2int_rn(data[threadId] + (offset) + 0.01f), 255.0f), 0.0f);
}

/*
* Transforms the image taken into the proper data type.
*/
__global__ void removeOffsetAndApplyMaxMin(int *data, int bitDepth, int signedOrUnsigned)
{
	int offset = 0;
	if (signedOrUnsigned == 0)
		offset = 1 << (bitDepth - 1);

	int threadId = threadIdx.x + blockIdx.x * blockDim.x;
	//0.01f used to avoid cases in which CUDA is not rounding correctly x.5 values. For example, 4.5 is rounded to 4 instead of 5, but 4.51 is correctly rounded to 5.
	data[threadId] = max(min(data[threadId] + (offset), 255), 0);
}

/*
* Function which manages the general flow of processing instruction to decode images, either in lossy/lossless or grayscale/RGB.
*/
void DecodingEngine::runImage()
{
	if (_waveletType == LOSSLESS)
	{
		if (_frameStructure->getIsRGB())
		{
			std::future<bool> readThread;
			readRGBCompressedBitStream();
			int bufferValue = 0;
			auto startProcessDisregardAllocationTimings = std::chrono::steady_clock::now();
			for (int i = 0; i < 3; i++)
			{
				bufferValue = _bufferReadingValueRGB[i];
				while (bufferValue == 0)
				{
					bufferValue = _bufferReadingValueRGB[i];
				}
				GPU_HANDLE_ERROR(cudaMemcpyAsync(_DBitStreamValues, _HBitStreamValuesRGB[i], _componentSizes[i] * sizeof(unsigned short), cudaMemcpyHostToDevice, cudaStreamDefault));
				SupportFunctions::markInitProfilerCPUSection("BPC", "BPC - Decoding");
				BPCCuda<unsigned short>* BPC = new BPCCuda<unsigned short>(_frameStructure, _HBitStreamValuesRGB[i], _waveletLevels, _DWTCBWidth, _DWTCBHeight, _codingPasses, _waveletType, _quantizationSize, _k, _LUTAmountOfBitplaneFiles);

				BPC->Decode(_frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight(), _LUTNumberOfBitplanes, _LUTNumberOfSubbands, _LUTContextRefinement, _LUTContextSign, _LUTContextSignificance, _LUTMultPrecision, _LUTInformation[i], _DPrefixedArray, _DSizeArray, _HBasicInformation, _DTempStoragePArray, _DBitStreamValues, _DCodeStreamValues, _HSizeArray, _HTotalBSSize, _DImagePixelsRGBTransformed[i] + _extraWaveletAllocation, cudaStreamDefault, _HLUTBSTableSteps, _DLUTBSTable, &_measurementsBPC[0]);
				
				SupportFunctions::markEndProfilerCPUSection();
			}
			SupportFunctions::markInitProfilerCPUSection("ColorTransform", "Color Tranform Kernel");
			prepareRGBImage(cudaStreamDefault);
			SupportFunctions::markEndProfilerCPUSection();
			auto finishProcessDisregardAllocationTimings = std::chrono::steady_clock::now();
			double elapsedTimeProcessDisregardAllocationTimings = std::chrono::duration_cast<std::chrono::duration<double>>(finishProcessDisregardAllocationTimings - startProcessDisregardAllocationTimings).count();
			std::cout << "The time spent with the app without considering allocation periods and I/O is: " << elapsedTimeProcessDisregardAllocationTimings << std::endl;
			writeRGBImage();
			std::cout << "BPC acum time is: " << _measurementsBPC[0] << std::endl;
		}
	}
	
}

/*
* General function which initializes the CPU threads needed depending on the amount of streams running the decoding.
*/
void DecodingEngine::engineManager(int cType)
{
	if (cType == IMAGE)
	{
		_measurementsBPC = new double[1];
		initMemory(cType);
		initLUT();
		runImage();
	}
}