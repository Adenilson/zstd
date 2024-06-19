/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
**/

/* This example deals with Dictionary compression,
 * its counterpart is `examples/dictionary_decompression.c` .
 * These examples presume that a dictionary already exists.
 * The main method to create a dictionary is `zstd --train`,
 * look at the CLI documentation for details.
 * Another possible method is to employ dictionary training API,
 * published in `lib/zdict.h` .
**/

#include <stdio.h>     // printf
#include <stdlib.h>    // free
#include <string.h>    // memset, strcat

#define ZDICT_STATIC_LINKING_ONLY

#include <zstd.h>      // presumes zstd library is installed

#include "common.h"    // Helper functions, CHECK(), and CHECK_ZSTD()

#include <stdint.h>
#include <zdict.h>
#include <pthread.h>


static void populate_buffer(char *buffer, size_t sampleSize, size_t samples)
{
    fprintf(stderr, "##### %s\n", __func__);
    const size_t elements = sampleSize * samples;
    for (size_t i = 0; i < elements; ++i) {
	buffer[i] = i & 0xff;
    }

}

static void populate_buffer_random(char *buffer, size_t sampleSize, size_t samples)
{
    fprintf(stderr, "##### %s\n", __func__);
    for (size_t i = 0; i < samples; ++i) {
	for (size_t j = 0; j < sampleSize; ++j) {
	    if (rand() % 2)
		buffer[i * sampleSize + j] = j & 0xff;
	    else
		buffer[i * sampleSize + j] = j;
	}
    }
}

static void* train_dictionary(void *arg)
{
    fprintf(stderr, "### %s: start\n", __func__);
    void* srcBuffer = NULL;
    ZSTD_CCtx* const cctx = ZSTD_createCCtx();
    size_t dictSize = 112000;
    void* dictBuffer = malloc(dictSize);
    size_t const sampleSize = 100000;
    uint32_t const nbSamples = 160;
    size_t* const samplesSizes = (size_t*) malloc(nbSamples * sizeof(size_t));
    ZDICT_cover_params_t params;
    uint32_t dictID;
    const size_t elements = sampleSize * nbSamples;
    if (!(srcBuffer = malloc(elements * sizeof(char))))
        goto _output_error;

    if (srcBuffer==NULL || dictBuffer==NULL || samplesSizes==NULL) {
	goto _output_error;
    }

    if (rand() % 2)
	populate_buffer_random((char*)srcBuffer, sampleSize, nbSamples);
    else
	populate_buffer((char*)srcBuffer, sampleSize, nbSamples);


    for (uint32_t u = 0; u < nbSamples; u++)
        samplesSizes[u] = sampleSize;

    memset(&params, 0, sizeof(params));
    params.k = 50;
    params.d = 6;
    params.nbThreads = 1;
    params.splitPoint = 1;

    dictSize = ZDICT_trainFromBuffer_cover(dictBuffer, dictSize,
					   srcBuffer, samplesSizes, nbSamples,
					   params);

    fprintf(stderr, "### %s: called trainFromBuffer_cover.\n", __func__);
    if (ZDICT_isError(dictSize))  {
      fprintf(stderr, "### %s: failed....", __func__);
      goto _output_error;
    }

    printf("OK, created dictionary of size %u \n", (unsigned)dictSize);

    dictID = ZDICT_getDictID(dictBuffer, dictSize);
    if (dictID==0) goto _output_error;
    printf("OK : %u \n", (unsigned)dictID);


_output_error:
    fprintf(stderr, "### %s: done!\n", __func__);
    ZSTD_freeCCtx(cctx);
    free(srcBuffer);
    free(dictBuffer);
    free(samplesSizes);

    return NULL;
}

static void crash_and_burn()
{
    const static size_t threads = 48;
    pthread_t t[threads];

    int res = 0;
    for (size_t i = 0; i < threads; ++i) {
	res = pthread_create(&t[i], NULL, &train_dictionary, NULL);
	if (res != 0) {
	    fprintf(stderr, "### Failed at thread creation. Exiting...\n");
	    goto exit;
	}
    }

    for (size_t i = 0; i < threads; ++i) {
	void *output = NULL;
	res = pthread_join(t[i], &output);
	if (res != 0) {
	    fprintf(stderr, "### Failed at thread joining: thread: %lu\n", i);
	}
    }

exit:
    fprintf(stderr, "#### Done.\n");
}

int main(int argc, const char** argv)
{
    crash_and_burn();
    return 0;
}
