

#if 0
#include "cstdio"
#include "iostream"
#include "string"
#include "fstream"

using namespace std;

#define MIN_(x,y) (x)>(y)?(y):(x)

int GetProfileString(string file_name, string section_name, string item_name,
        string &item_value)
{
    ifstream mystream ;
    mystream.open(file_name.c_str(), ios::in);

    if (!mystream)
    {
        cout << "Error " << endl;
        return -1;
    }

    char line[30] = {0};
    string line2;
    string::size_type return_of_find,find1, find2,end1 = 0,end2 = 0;
    bool found = false;

    while (mystream.getline(line, 30) && !found) //默认行不会超过30个字符
    {
        line2 = line;

        return_of_find = line2.find(section_name);
        if (string::npos == return_of_find)
            continue; //没找到section项，则继续下⼀行读取

//找到了，则执行第二步，寻找相应的键值，关键是不能跨越多段
        while (mystream.getline(line, 30) && !found)
        {
            line2 = line;
            string equal_flag = "=";

            return_of_find = line2.find(equal_flag);
            if (string::npos == return_of_find)
                return -1;//说明已经跨越了多段，目标寻找失败
//还在当前段中
            return_of_find = line2.find(item_name);
            if (string::npos == return_of_find)
                continue;    //没有找到
//找到了
            // return_of_find = line2.rfind(" "); //要求配置文件=两边要有空格
		return_of_find = line2.find(equal_flag);

/*
		find1 = line2.find(" ");
		find2 = line2.find("#");
		if(find1 != string::npos && find1 > return_of_find)
			end1 = find1;

		if(find2 != string::npos && find2 > return_of_find)
			end2 = find2;		
		
		find1= MIN_(end1,end2);
		if(find1 < )
*/
            item_value = line2.substr(return_of_find + 1); //该行最后一个空格之后开始的为所要的item_value
            found = true;
        }
    }
    mystream.close();
    return 0;
}

int main()
{
	string str;
	GetProfileString("config","[face_config]","MXFC_GPIO_DELAY",str);
	cout<<str;
	return 0;
}


#endif




#include <fstream>
#include <iostream>
#include <stdlib.h>

#include <string>
#include <map>
using namespace std;
 
#define COMMENT_CHAR '#'



bool IsSpace(char c)
{
    if (' ' == c || '\t' == c)
        return true;
    return false;
}

bool IsCommentChar(char c)
{
    switch(c) {
    case COMMENT_CHAR:
        return true;
    default:
        return false;
    }
}

void Trim(string & str)
{
    if (str.empty()) {
        return;
    }
    int i, start_pos, end_pos;
    for (i = 0; i < str.size(); ++i) {
        if (!IsSpace(str[i])) {
            break;
        }
    }
    if (i == str.size()) { // 全部是空白字符串
        str = "";
        return;
    }
    
    start_pos = i;
    
    for (i = str.size() - 1; i >= 0; --i) {
        if (!IsSpace(str[i])) {
            break;
        }
    }
    end_pos = i;
    
    str = str.substr(start_pos, end_pos - start_pos + 1);
}

bool AnalyseLine(const string & line, string & key, string & value)
{
    if (line.empty())
        return false;
    int start_pos = 0, end_pos = line.size() - 1, pos;
    if ((pos = line.find(COMMENT_CHAR)) != -1) {
        if (0 == pos) {  // 行的第一个字符就是注释字符
            return false;
        }
        end_pos = pos - 1;
    }
    string new_line = line.substr(start_pos, start_pos + 1 - end_pos);  // 预处理，删除注释部分
    
    if ((pos = new_line.find('=')) == -1)
        return false;  // 没有=号
        
    key = new_line.substr(0, pos);
    value = new_line.substr(pos + 1, end_pos + 1- (pos + 1));
    
    Trim(key);
    if (key.empty()) {
        return false;
    }
    Trim(value);
    return true;
}

bool ReadConfig(const string & filename, map<string, string> & m)
{
    m.clear();
    ifstream infile(filename.c_str());
    if (!infile) {
        cout << "file open error" << endl;
        return false;
    }
    string line, key, value;
    while (getline(infile, line)) {
        if (AnalyseLine(line, key, value)) {
            m[key] = value;
        }
    }
    
    infile.close();
    return true;
}

int GetConfig(const map<string, string> & m, string key_name,string &key)
{
    map<string, string>::const_iterator mite = m.begin();
    for (; mite != m.end(); ++mite) {
        //cout << mite->first << "=" << mite->second << endl;
        if(key_name.compare(mite->first) == 0){
            //cout << "find key "<<key_name<<":"<<mite->second<< endl;
            key = mite->second;
            return 0;
        }
    }
    return -1;
}

#if 0
int main()
{
    map<string, string> m;
	string key("MXFC_GPIO_DELAY");
	string key_val;

    ReadConfig("config", m);
    //PrintConfig(m);
	GetConfig(m,"MXFC_GPIO_DELAY",key_val);
	cout<<key_val;
    
    int a=atoi(key_val.c_str());
    printf("a =%d\n",a);
     
    return 0;
}

#endif
