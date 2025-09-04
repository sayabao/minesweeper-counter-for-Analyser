# 使用方法
截图，保存为map.png，运行 main.py ，获得 map.mine ，用于JSMinesweeper的Minesweeper analyser
# 截图示例
![一个范例](/map.png)
# 注
非常简单粗暴的识别颜色判断数字，所以界面皮肤不同会导致无法使用

目前仅适用于Wom的Classic(dark)皮肤

而且暂时无法识别7和8~~等遇到了再加上~~

~~而且暂时只适用于高级（16×30×99）~~
添加了适用于中级以及自定义的.py

自定义需要自个手动修改行列数和最后的总雷数

>num_rows, num_cols = `28`, `21`
>
>output_str = f"{num_cols}x{num_rows}x`140`\n" + "\n".join("".join(row) for row in matrix)
