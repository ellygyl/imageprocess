# 任务目标
请帮我按照如下具体需求实现一个实现图像相关任务的skill

# 技能实现说明
1. 目标检测：基于yolov5或者其他的模型进行目标检测，输入多张图片和目标说明，从图片把目标框检测返回
    - 输入：x张图片和目标说明
    - 输出：json格式的左上角、右下角坐标
    - 示例：
      - 输入：图片地址 https://pics7.baidu.com/feed/09fa513d269759ee183c20de87d927196d22df6c.jpeg@f_auto?token=47985a0a0ad265709ff4ee6d44ed7b9b ，描述：检测图片中有无猫
      - 输出：{"0":[100,200,300,400]}
  2. 图像分类：基于resnet50，给定图片，判断图片中包含什么内容
    - 输入：图片地址https://pics7.baidu.com/feed/09fa513d269759ee183c20de87d927196d22df6c.jpeg@f_auto?token=47985a0a0ad265709ff4ee6d44ed7b9b，文本：是否有猫
    - 输出：是的，有猫和狗；没有，没有猫和狗
  3. 语义分割：给定图片和区域，判断该区域内包含什么内容
    - 输入：图片地址https://pics7.baidu.com/feed/09fa513d269759ee183c20de87d927196d22df6c.jpeg@f
    - 输出：语义分隔的结果

# 限制条件：
1. skill的名称自定义
2. skill的脚本使用python编写
3. 代码放到/Users/guyanlei/Desktop/图像处理/这个目录下
4. 代码中可以使用模型外部库，但需要给出安装命令

# 技能返回格式
请按照如下格式返回：
{
    "result": "0.0",
    "error": "0.0",
    "answer": ""
}
请直接返回json，不要返回其他无关信息
