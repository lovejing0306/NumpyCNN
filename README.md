# NumpyCNN

使用纯 NumPy 代码构建构建卷积神经网络

卷积神经网络（CNN）是分析图像等多维信号的当前最优技术。目前已有很多库可以实现 CNN，如 TensorFlow 和 Keras 等。
这种库仅提供一个抽象的 API，因此可以大大降低开发难度，并避免实现的复杂性，不过使用这种库的开发人员无法接触到一些细节，这些细节可能在实践中很重要。

有时，数据科学家必须仔细查看这些细节才能提高性能。在这种情况下，最好自己亲手构建此类模型，这可以帮助你最大程度地控制网络。
因此在本文中，我们将仅使用 NumPy 尝试创建 CNN。我们会创建三个层，即卷积层（简称 conv）、ReLU 层和最大池化层。所涉及的主要步骤如下：
1. 读取输入图像
2. 准备滤波器
3. 卷积层：使用滤波器对输入图像执行卷积操作
4. ReLU 层：将 ReLU 激活函数应用于特征图（卷积层的输出）
5. 最大池化层：在 ReLU 层的输出上应用池化操作
6. 堆叠卷积层、ReLU 层和最大池化层


## 参考链接
[ahmedfgad的github][1]

[1]: https://github.com/ahmedfgad/NumPyCNN