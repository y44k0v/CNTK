//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// RNGHandle.h: An abstraction around a random number generator
//

#pragma once

#include "Constants.h"
#include "CommonMatrix.h"
#include "File.h"
#include <memory>

#ifndef CNTK_MODEL_VERSION_16
#define CNTK_MODEL_VERSION_16 16
#endif


namespace Microsoft { namespace MSR { namespace CNTK {

class MATH_API RNGHandle
{
public:
    static std::shared_ptr<RNGHandle> Create(DEVICEID_TYPE deviceId, uint64_t seed, uint64_t offset = 0);
    
    virtual ~RNGHandle() {}

    DEVICEID_TYPE DeviceId() const
    {
        return m_deviceId;
    }

protected:
    RNGHandle(DEVICEID_TYPE deviceId)
        : m_deviceId(deviceId)
    {}

private:

    DEVICEID_TYPE m_deviceId;
};

// Nodes using a random number generators should derive from this interface.
// One purpose of this interface is to have a common interface for setting the seeds when setting up a network.
class IRngUser
{
public:
    virtual RNGHandle& GetRNGHandle(DEVICEID_TYPE deviceId) = 0;
    virtual void SetRngState(uint64_t seed, uint64_t offset = 0) = 0;
};

// This implements IRngUser using RNGHandle.
class RngUser : public IRngUser
{
public:
    RNGHandle& GetRNGHandle(DEVICEID_TYPE deviceId) override
    {
        if (!m_RNGHandle)
            m_RNGHandle = RNGHandle::Create(deviceId, m_rngSeed, m_rngOffset);

        return *m_RNGHandle;
    }

    // E.g. called from ComputationNetwork to make sure that CNTK running on different nodes will have different seed.
    void SetRngState(uint64_t seed, uint64_t offset = 0) override
    {
        m_rngSeed = seed;
        m_rngOffset = offset;
        m_RNGHandle.reset(); // Reset handle. New handle will be generated with next call of GetRNGHandle(...).
    }

    uint64_t GetRngSeed() const
    {
        return m_rngSeed;
    }

    uint64_t GetRngOffset() const
    {
        return m_rngOffset;
    }

    void UpdateRngOffset(uint64_t val)
    {
        m_rngOffset = val;
    }

protected:

    void Load(File& fstream, size_t modelVersion)
    {
        if (modelVersion < CNTK_MODEL_VERSION_16)
            return;

        uint64_t seed;
        uint64_t offset;

        if (modelVersion == CNTK_MODEL_VERSION_16)
        {
            unsigned long seed_16;
            fstream >> seed_16;
            seed = seed_16;
        }
        else
        {
            fstream >> seed;
        }

        fstream >> offset;
        SetRngState(seed, offset);
    }

    void Save(File& fstream) const
    {
        fstream << GetRngSeed();
        fstream << GetRngOffset();
    }

    uint64_t m_rngSeed = 0;
    uint64_t m_rngOffset = 0;
    std::shared_ptr<RNGHandle> m_RNGHandle;
};


}}}
