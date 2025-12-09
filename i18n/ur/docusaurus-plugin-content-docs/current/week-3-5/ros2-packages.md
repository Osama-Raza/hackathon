---
title: "ROS 2 پیکیجز تعمیر کرنا"
sidebar_label: "ROS 2 پیکیجز تعمیر کرنا"
description: "CMakeLists.txt اور package.xml کے ساتھ ROS 2 پیکیجز کیسے بنائیں اور ساخت دیں سیکھیں"
sidebar_position: 2
---

# ROS 2 پیکیجز تعمیر کرنا

## ROS 2 پیکیج کیا ہے؟

پیکیج ROS 2 سسٹم کا بنیادی تعمیر کا بلاک ہے۔ یہ نوڈز، لائبریریز، کنفیگریشن فائلیں، اور دیگر وسائل پر مشتمل ہوتا ہے جو کہ مخصوص فعالیت کے لیے ضروری ہوتے ہیں۔ پیکیجز ROS 2 کوڈ کو منظم کرنے اور تقسیم کرنے کا طریقہ فراہم کرتے ہیں۔

### کلیدی خصوصیات

- **ماڈولر**: ہر پیکیج ایک مخصوص کام انجام دیتا ہے
- **قابلِ تقسیم**: دوسروں کے ساتھ شیئر کیا جا سکتا ہے
- **قابلِ دوبارہ استعمال**: دوسرے پروجیکٹس میں استعمال کیا جا سکتا ہے
- **آزاد**: دوسرے پیکیجز سے آزاد ہوتا ہے (ضروری ڈیپینڈنسیز کے علاوہ)

## ROS 2 پیکیج کی ساخت

ROS 2 پیکیجز مندرجہ ذیل ساخت کا پیروی کرتے ہیں:

```
my_robot_package/           # پیکیج کا نام
├── CMakeLists.txt         # C++/C بلڈ کنفیگریشن
├── package.xml            # پیکیج کا میٹا ڈیٹا
├── src/                   # C++ سورس کوڈ
│   ├── node1.cpp
│   └── node2.cpp
├── include/               # C++ ہیڈر فائلیں
│   └── my_robot_package/
│       ├── node1.hpp
│       └── node2.hpp
├── scripts/               # پائی تھون اسکرپٹس
├── launch/                # ROS 2 لانچ فائلیں
│   ├── node1_launch.py
│   └── node2_launch.py
├── config/                # کنفیگریشن فائلیں
│   └── params.yaml
├── test/                  # ٹیسٹ کوڈ
└── msg/                   # کسٹم پیغامات (اگر کوئی ہو)
    └── CustomMessage.msg
```

## package.xml کی ترتیب

`package.xml` فائل پیکیج کا میٹا ڈیٹا فراہم کرتی ہے:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>1.0.0</version>
  <description>My robot functionality package</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache License 2.0</license>

  <depend>rclcpp</depend>
  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>

  <exec_depend>ros2launch</exec_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

## CMakeLists.txt کی ترتیب

`CMakeLists.txt` فائل C++ کوڈ کی بلڈ کنفیگریشن فراہم کرتی ہے:

```cmake
cmake_minimum_required(VERSION 3.8)
project(my_robot_package)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# نوڈ تعریف کریں
add_executable(my_node src/my_node.cpp)
ament_target_dependencies(my_node rclcpp std_msgs)

# نوڈ کو انسٹال کریں
install(TARGETS
  my_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
```

## پائی تھون پیکیج کی مثال

اگر آپ پائی تھون کوڈ لکھ رہے ہیں تو آپ کو `setup.py` فائل کی ضرورت ہوگی:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'my_robot_package'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # لانچ فائلیں شامل کریں
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='user@example.com',
    description='My robot functionality package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'my_node = my_robot_package.my_node:main',
        ],
    },
)
```

## کمپائل کرنا اور انسٹال کرنا

### 1. ورک سپیس تیار کریں

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

### 2. اپنا پیکیج تیار کریں

```bash
ros2 pkg create --build-type ament_cmake my_robot_package
```

### 3. کوڈ لکھیں

- `src/` میں C++ کوڈ شامل کریں
- `scripts/` میں پائی تھون اسکرپٹس شامل کریں
- `launch/` میں لانچ فائلیں شامل کریں

### 4. بلڈ کریں

```bash
cd ~/ros2_ws
colcon build --packages-select my_robot_package
source install/setup.bash
```

## ROS 2 پیکیجز کی اقسام

### 1. نوڈ پیکیجز
- واحد یا متعدد ROS 2 نوڈز فراہم کرتے ہیں
- مخصوص کام کرتے ہیں (جیسے: نیوی گیشن، سینسنگ، کنٹرول)

### 2. لائبریری پیکیجز
- دیگر پیکیجز کے لیے کوڈ فراہم کرتے ہیں
- کوئی نوڈ نہیں ہوتا
- دوبارہ استعمال کے لیے ڈیزائن کیا گیا

### 3. میٹا پیکیجز
- دیگر پیکیجز کا مجموعہ ہوتا ہے
- کوئی کوڈ نہیں ہوتا
- کوائف کے انسٹالیشن کو آسان بناتا ہے

### 4. کانفیگریشن پیکیجز
- صرف کنفیگریشن فائلیں فراہم کرتے ہیں
- کوئی کوڈ نہیں ہوتا
- مختلف ماحول کی ترتیبات کے لیے

## بہترین طریقے

### 1. مناسب نامزدگی
- واضح اور تفصیلی نام استعمال کریں
- snake_case استعمال کریں (جیسے: my_robot_navigation)
- ROS 2 کے معیاری ناموں سے بچیں

### 2. ڈیپینڈنسیز کا انتظام
- صرف ضروری ڈیپینڈنسیز شامل کریں
- `package.xml` میں ہر ڈیپینڈنسی کی وضاحت کریں
- `CMakeLists.txt` میں ہر ڈیپینڈنسی کو تلاش کریں

### 3. ٹیسٹنگ
- ہر فعالیت کے لیے یونٹ ٹیسٹس لکھیں
- ٹیسٹس کو `test/` ڈائریکٹری میں رکھیں
- `colcon test` کا استعمال کرکے ٹیسٹس چلائیں

### 4. دستاویزات
- ہر فعالیت کی وضاحت کریں
- استعمال کی مثالیں فراہم کریں
- README.md فائل شامل کریں

ROS 2 پیکیجز کو سمجھنا اور تعمیر کرنا ROS 2 میں کامیابی کی کلید ہے۔ یہ آپ کو ایک ماڈولر، دوبارہ استعمال کے قابل، اور منظم کوڈ لکھنے کی اجازت دیتا ہے۔

## پیکیج کی ساخت

ROS 2 کا ایک معمولی پیکیج مندرجہ ذیل ساخت رکھتا ہے:

```
my_package/
├── CMakeLists.txt          # C++ کے لیے بلڈ کنفیگریشن
├── package.xml             # پیکیج کا مظہر نامہ
├── src/                    # سورس کوڈ فائلیں
│   ├── publisher_node.cpp
│   └── subscriber_node.cpp
├── include/my_package/     # ہیڈر فائلیں
├── launch/                 # لانچ فائلیں
├── config/                 # کنفیگریشن فائلیں
├── test/                   # یونٹ اور انٹیگریشن ٹیسٹس
├── scripts/                # پائی تھون اسکرپٹس
├── msg/                    # کسٹم پیغام کی تعریفات
├── srv/                    # کسٹم سروس کی تعریفات
└── action/                 # کسٹم ایکشن کی تعریفات
```

## نیا پیکیج تخلیق کرنا

### colcon کا استعمال کریں

نیا پیکیج تخلیق کرنے کا تجویز کردہ طریقہ `ros2 pkg create` کمانڈ کا استعمال کرنا ہے:

```bash
ros2 pkg create --build-type ament_cmake my_robot_package
```

پائی تھون پیکیجز کے لیے:
```bash
ros2 pkg create --build-type ament_python my_robot_package
```

### پیکیج کے اختیارات

پیکیج تخلیق کرتے وقت آپ اضافی اختیارات متعین کر سکتے ہیں:

```bash
# ڈیپینڈنسیز کے ساتھ تخلیق کریں
ros2 pkg create --build-type ament_cmake --dependencies rclcpp rclpy std_msgs my_robot_package

# نوڈز کے ساتھ تخلیق کریں
ros2 pkg create --build-type ament_cmake --node-name my_node my_robot_package

# کسٹم مینٹینر کی معلومات کے ساتھ تخلیق کریں
ros2 pkg create --build-type ament_cmake --maintainer-email "user@example.com" --maintainer-name "Your Name" my_robot_package
```

## package.xml: پیکیج کا مظہر نامہ

`package.xml` فائل پیکیج کے بارے میں میٹا ڈیٹا پر مشتمل ہوتی ہے:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>0.0.0</version>
  <description>میرے روبوٹ کے لیے ROS 2 کا مثالی پیکیج</description>
  <maintainer email="user@example.com">آپ کا نام</maintainer>
  <license>Apache لائسنس 2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <depend>rclcpp</depend>
  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

### package.xml میں کلیدی عناصر

- **name**: پیکیج کا نام (آپ کے ورک اسپیس میں منفرد ہونا چاہیے)
- **version**: سیمینٹک ورژننگ (MAJOR.MINOR.PATCH)
- **description**: پیکیج کی مختصر تفصیل
- **maintainer**: پیکیج مینٹینر کے لیے رابطے کی معلومات
- **license**: سافٹ ویئر لائسنس کی معلومات
- **buildtool_depend**: استعمال شدہ بلڈ سسٹم (ament_cmake، ament_python)
- **depend**: رن ٹائم ڈیپینڈنسیز
- **build_depend**: بلڈ ٹائم ڈیپینڈنسیز
- **exec_depend**: ایگزیکیوشن ٹائم ڈیپینڈنسیز

## CMakeLists.txt: بلڈ کنفیگریشن

C++ پیکیجز کے لیے، `CMakeLists.txt` فائل یہ بیان کرتی ہے کہ پیکیج کیسے بنایا جاتا ہے:

```cmake
cmake_minimum_required(VERSION 3.8)
project(my_robot_package)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ڈیپینڈنسیز تلاش کریں
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)

# قابلِ ایگزیکیوشن شامل کریں
add_executable(talker src/talker.cpp)
ament_target_dependencies(talker rclcpp std_msgs)

add_executable(listener src/listener.cpp)
ament_target_dependencies(listener rclcpp std_msgs)

# ایگزیکیوٹیبلز انسٹال کریں
install(TARGETS
  talker
  listener
  DESTINATION lib/${PROJECT_NAME}
)

# دیگر فائلیں انسٹال کریں
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

# ٹیسٹ کنفیگریشن
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# پیکیج کنفیگریشن
ament_package()
```

### کلیدی CMake ہدایات

- **cmake_minimum_required**: ضروری CMake ورژن کا کم از کم
- **project**: پراجیکٹ کا نام تعریف کریں
- **find_package**: دیگر پیکیجز تلاش کریں اور لوڈ کریں
- **add_executable**: قابلِ ایگزیکیوشن ہدف بنائیں
- **ament_target_dependencies**: ہدف کے لیے ڈیپینڈنسیز متعین کریں
- **install**: انسٹال کرنے کے لیے فائلیں متعین کریں
- **ament_package**: پیکیج کنفیگریشن مکمل کریں

## پائی تھون پیکیج کی ساخت

پائی تھون پیکیجز کے لیے، ساخت میں تھوڑا فرق ہوتا ہے:

```
my_python_package/
├── setup.py                # پائی تھون سیٹ اپ کنفیگریشن
├── package.xml             # ROS 2 پیکیج کا مظہر نامہ
├── my_python_package/      # پائی تھون ماڈیول
│   ├── __init__.py
│   └── my_module.py
└── test/
```

پائی تھون پیکیجز کے لیے `setup.py` فائل:

```python
from setuptools import setup

package_name = 'my_python_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='آپ کا نام',
    maintainer_email='user@example.com',
    description='ROS 2 کا پائی تھون کی مثال',
    license='Apache لائسنس 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'my_node = my_python_package.my_node:main',
        ],
    },
)
```

## اپنا پیکیج بنانا

### colcon کا استعمال کریں

اپنا پیکیج بنانے کے لیے:

```bash
# ورک اسپیس میں جائیں
cd ~/ros2_ws

# مخصوص پیکیج کو بلڈ کریں
colcon build --packages-select my_robot_package

# تمام پیکیجز کو بلڈ کریں (ڈیولپمنٹ کے لیے سیmlink کے ساتھ)
colcon build --symlink-install

# مخصوص اختیارات کے ساتھ بلڈ کریں
colcon build --packages-select my_robot_package --cmake-args -DCMAKE_BUILD_TYPE=Release
```

### ماحول کو سورس کرنا

بلڈ کے بعد، آپ کو سیٹ اپ فائلیں سورس کرنے کی ضرورت ہے:

```bash
# ورک اسپیس کو سورس کریں
source install/setup.bash

# یا دائمی سروش کے لیے اپنے .bashrc میں شامل کریں
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

## عام بلڈ مسائل اور حل

### ڈیپینڈنسیز کی کمی

اگر آپ کو غیر موجود ڈیپینڈنسیز کی وجہ سے بلڈ کی خرابیوں کا سامنا ہوتا ہے:

```bash
# غیر موجود ROS پیکیجز انسٹال کریں
sudo apt update
sudo apt install ros-humble-<package-name>

# یا متعدد پیکیجز انسٹال کریں
sudo apt install ros-humble-rclcpp ros-humble-std-msgs ros-humble-sensor-msgs
```

### اجازت کے مسائل

اگر آپ کو اجازت کے مسائل کا سامنا ہوتا ہے:

```bash
# اپنے ورک اسپیس پر اجازتیں چیک کریں
ls -la ~/ros2_ws/

# یقینی بنائیں کہ فائلیں آپ کے مالکانہ ہیں
sudo chown -R $USER:$USER ~/ros2_ws/
```

### صاف بلڈ

اگر آپ کو مسلسل بلڈ کے مسائل کا سامنا ہوتا ہے:

```bash
# بلڈ آرٹیفیکٹس صاف کریں
rm -rf build/ install/ log/

# دوبارہ بلڈ کریں
colcon build --symlink-install
```

## بہترین طریقے

1. **وضاحتی نام**: پیکیجز کے لیے صاف، وضاحتی نام استعمال کریں
2. ** واحد ذمہ داری**: ہر پیکیج کو ایک واحد، اچھی طرح سے وضاحت شدہ مقصد ہونا چاہیے
3. **ڈیپینڈنسی مینجمنٹ**: صرف ضروری ڈیپینڈنسیز شامل کریں
4. **ورژن کنٹرول**: Git کا استعمال کریں .gitignore فائلیں کے ساتھ
5. **دستاویزات**: README فائلیں اور ان لائن دستاویزات شامل کریں
6. **ٹیسٹنگ**: اہم فعالیت کے لیے یونٹ ٹیسٹس شامل کریں
7. **مطابقت کی ساخت**: ROS 2 پیکیج کنونشن کا پیروی کریں

## مثال: مکمل پیکیج کا عمل

آئیے ایک سادہ پبلشر/سبسکرائیبر کی مثال بنائیں:

1. پیکیج تخلیق کریں:
```bash
ros2 pkg create --build-type ament_cmake --dependencies rclcpp rclpy std_msgs example_talker
```

2. `src/` ڈائریکٹری میں سورس فائلیں شامل کریں
3. `CMakeLists.txt` اور `package.xml` کو اپ ڈیٹ کریں
4. پیکیج کو بلڈ کریں: `colcon build --packages-select example_talker`
5. ماحول کو سورس کریں: `source install/setup.bash`
6. نوڈز چلائیں: `ros2 run example_talker talker` اور `ros2 run example_talker listener`

ROS 2 پیکیجز کو مناسب طریقے سے ساخت اور بلڈ کرنے کے طریقے کو سمجھنا ماڈیولر، قابلِ دیکھ بھال روبوٹکس ایپلی کیشنز تیار کرنے کے لیے ضروری ہے۔ یہ بنیاد آپ کو اچھی طرح سے منظم کوڈ تخلیق کرنے میں مدد دے گی جو ROS 2 کے وسیع ایکو سسٹم کے ساتھ بے رکاوٹ اندراج کرے گا۔