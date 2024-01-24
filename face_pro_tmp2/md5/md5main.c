/*
  Copyright (C) 2002 Aladdin Enterprises.  All rights reserved.

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  L. Peter Deutsch
  ghost@aladdin.com

 */
/* $Id: md5main.c,v 1.1 2002/04/13 19:20:28 lpd Exp $ */
/*
  Independent implementation of MD5 (RFC 1321).

  This code implements the MD5 Algorithm defined in RFC 1321, whose
  text is available at
	http://www.ietf.org/rfc/rfc1321.txt
  The code is derived from the text of the RFC, including the test suite
  (section A.5) but excluding the rest of Appendix A.  It does not include
  any code or documentation that is identified in the RFC as being
  copyrighted.

  The original and principal author of md5.c is L. Peter Deutsch
  <ghost@aladdin.com>.  Other authors are noted in the change history
  that follows (in reverse chronological order):

  2002-04-13 lpd Splits off main program into a separate file, md5main.c.
 */


#include <stdio.h>
#include <string.h>
#include "md5.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>


int main()
{
   int i = 0;
   int fd; char buf[4096]; int ret;
   
    md5_byte_t digest[16];
    md5_state_t md5StateT;
    char md5String[33] = { '\0' }, hexBuffer[3];
    
    //md5值获取
    md5_init(&md5StateT);

     
    fd = open("./md5.h",O_RDONLY); 
    if(fd < 0) { 
        printf("open file fail\n"); 
        return 0;
    } 

  //md5值获取，md5_init开头已经执行
    while((ret = read(fd,buf,sizeof(buf)) <= 0))
    {
        md5_append(&md5StateT,buf,ret);
    }
    md5_finish(&md5StateT, digest);
    

    for(i = 0;i<16;i++)
    {
        snprintf(md5String+i*2,3,"%02X",digest[i]);
    }
    md5String[32] = '\0';
    printf("the file md5 is %s\n",md5String);

    close(fd);


    
    md5_init(&md5StateT);
    fd = open("./md5.h",O_RDONLY); 
    if(fd < 0) { 
        printf("open file fail\n"); 
        return 0;
    } 

  //md5值获取，md5_init开头已经执行
    while((ret = read(fd,buf,sizeof(buf)) <= 0))
    {
        md5_append(&md5StateT,buf,ret);
    }
    md5_finish(&md5StateT, digest);
    

    for(i = 0;i<16;i++)
    {
        snprintf(md5String+i*2,3,"%02X",digest[i]);
    }
    md5String[32] = '\0';
    printf("the file md5 is %s\n",md5String);

    close(fd);


    
    return 0;
}



