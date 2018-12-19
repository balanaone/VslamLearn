#ifndef CONFIG_H
#define CONFIG_H

#include "myslam/common_include.h"

namespace myslam
{

class Config
{
private:
    static std::shared_ptr<Config> config_;
    cv::FileStorage file_;

    Config() {} // private constructor makes singleton
public:
    ~Config();

    // set a new config file
    static void setParameterFile(const std::string & filename);

    // access teh parameter values
    template<typename T>
    static T get(const std::string& key)
    {
        return T(Config::config_->file_[key]);
    }
};

}

#endif // CONFIG_H

