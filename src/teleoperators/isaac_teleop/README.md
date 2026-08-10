暂时不需要home,移除home所有相关逻辑;

重构 get_action，我想要 get_action 逻辑如下:

pos 逻辑为按下squeeze后的相对位移: x 为左右(左-右+);y为前后(前+后-);z为高低(高+低-)

ori 逻辑为按下squeeze后的相对姿态偏移

例如: squeeze 没按过: pos = (0,0,0) 输出0相对位移

    按下 squeeze:          pos = (0,0,0)  记录相对位移的初始位置

    手柄往前推 10cm:       pos = (0,0.1,0)  相对初始位置的移动 (跟手)

    手柄往左移动 20cm:       pos = (-0.2,0,0)

    手柄往上抬 5cm:        pos = (0,0,0.05)

    松开 squeeze:          pos = (0,0,0)   (冻结)
