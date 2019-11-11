/*
 * SimpleConfig.cuh
 *
 *  Created on: 2018 aug. 29
 *      Author: ervin
 */

#ifndef SIMPLECONFIG_CUH_
#define SIMPLECONFIG_CUH_

#include <string>
#include <map>
#include <vector>

using namespace std;

// Config class

class SimpleConfig {
private:
	map<string, string> settings;
	vector<string> split(const string &s, char delim);
public:
	SimpleConfig();
	SimpleConfig(map<string, string>& default_settings);
	map<string, string> getSettings();
	void setSettings(map<string, string>& settings);
	void addFromString(string argument);
	void addFromConfigFile(string file_path);
	string getProperty(string propName);
	void setProperty(string propName, string value);
	float getFProperty(string propName);
	string getSProperty(string propName);
	int getIProperty(string propName);
    string& operator[] (const string propName);
    const string& operator[] (const string propName) const;
    void printSettings();
};

#endif /* SIMPLECONFIG_CUH_ */
