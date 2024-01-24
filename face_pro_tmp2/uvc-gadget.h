#ifndef __UVC_GADGET_H
#define __UVC_GADGET_H

#include "v4l2_test.h"

#ifdef __cplusplus
extern "C" 
{
#endif

int uvc_init(char *device,unsigned int max_packSize,struct V4L2_DEV *v4l2);
int uvc_event_loop(void );

#ifdef __cplusplus
}  /* end extern "C" */
#endif

#endif /* md5_INCLUDED */