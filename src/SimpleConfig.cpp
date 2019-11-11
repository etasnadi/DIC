#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <iterator>
#include <sstream>
#include <fstream>

#include "SimpleConfig.h"

using namespace std;

/* Splitting alg: https://stackoverflow.com/questions/236129/split-a-string-in-c */
template<typename Out>
void split(const string &s, char delim, Out result) {
	stringstream ss;
	ss.str(s);
	string item;
	while (std::getline(ss, item, delim)) {
		*(result++) = item;
	}
}

vector<string> split(const string &s, char delim) {
    vector<string> elems;
    ::split(s, delim, std::back_inserter(elems));
    return elems;
}

SimpleConfig::SimpleConfig(map<string, string>& defaultSettings){
	settings = defaultSettings;
}

SimpleConfig::SimpleConfig(){
}

void SimpleConfig::setSettings(map<string, string>& newSettings){
	settings = newSettings;
}

void parseConfigEntry(string entry, map<string, string>& settings){
	vector<string> setting = ::split(entry, '=');
	if(setting.size()>=2){
		if(setting[0][0] != '#'){
			settings[setting[0]] = setting[1];
		}
	}
}

void SimpleConfig::addFromString(string configString){
	vector<string> setting_vec = ::split(configString, ',');
	for(int i = 0; i < setting_vec.size(); i++){
		parseConfigEntry(setting_vec[i], settings);
	}
}

void SimpleConfig::addFromConfigFile(string configFilePath){
	std::ifstream input(configFilePath);
	string line;
	while(getline(input, line)){
		parseConfigEntry(line, settings);
	}
}

void SimpleConfig::setProperty(string propName, string propValue){
	settings[propName] = propValue;
}

string SimpleConfig::getProperty(string propName){
	return settings[propName];
}

float SimpleConfig::getFProperty(string propName){
	float result = atof(settings[propName].c_str());
	return result;
}

string SimpleConfig::getSProperty(string propName){
	string result = settings[propName];
	return result;
}

int SimpleConfig::getIProperty(string propName){
	int result = atoi(settings[propName].c_str());
	return result;
}

string& SimpleConfig::operator[] (const string propName){
	return settings[propName];
}

const string& SimpleConfig::operator[] (const string propName) const {
	return settings.at(propName);
}

void SimpleConfig::printSettings(){
	cout << "Current configuration: " << endl;
	for (auto const& setting : settings){
		cout << setting.first
				  << '='
				  << setting.second
				  << std::endl ;
	}
	cout << "=======================" << endl;
}
