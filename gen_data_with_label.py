gen_label_prompt = """
## 角色
你是一个医生，根据给出的病例数据做出对应的诊断。

## 任务
病人的病历如下所示：
```
{feature_content}
```
你需要根据当前病人的病历，给出诊断依据和可能的疾病。

## 输出要求
1. 以json格式进行输出，json包括3个字段分别是reason、diseases、feature_content；
2. reason为诊断依据；
3. diseases为可能的疾病；
4. 所输出的json，不需要使用```json进行标识，直接输出json；

## 示例
用户输入:
性别: 男\\n年龄: 65\\n主诉: 发热10天\\n现病史: 患者10天前无明显诱因开始出现发热，体温最高36.7℃，无咳嗽咳痰，无畏寒、寒战，无鼻塞流涕，无咽痛，无周身乏力，无嗅觉味觉减退，无胸闷、憋气,无头痛头晕，无恶心呕吐，无腹痛腹泻，无尿频、尿急、尿痛不适，于当地输液治疗，为求进一步治疗，来我院就诊。\\n既往史: 无\\n个人史: 有新型冠状病毒疫苗接种史，共3针。\\n过敏史: \\n婚育史: \\n流行病史: \\n体格检查: T36.08℃。\\n辅助检查: 
你的输出:
{{\"reason\":\"1. 患者主诉发热10天，但最高体温为36.7℃，体检时体温36.08℃，均在正常范围内，未达发热标准（≥37.3℃）；\\n2. 无咳嗽、咳痰、畏寒、寒战等呼吸道感染症状；无尿频、尿急、尿痛等泌尿系感染症状；无腹痛、腹泻等消化道症状；\\n3. 无全身乏力、体重减轻、盗汗等全身性症状；\\n4. 既往体健，无慢性疾病史；\\n5. 为明确患者自觉发热的原因，需进一步完善检查，如血常规、炎症指标、甲状腺功能等，排除感染、内分泌疾病或其他潜在病因。\",\"diseases\":\"发热\",\"feature_content\":\"性别: 男\\n年龄: 65\\n主诉: 发热10天\\n现病史: 患者10天前无明显诱因开始出现发热，体温最高36.7℃，无咳嗽咳痰，无畏寒、寒战，无鼻塞流涕，无咽痛，无周身乏力，无嗅觉味觉减退，无胸闷、憋气,无头痛头晕，无恶心呕吐，无腹痛腹泻，无尿频、尿急、尿痛不适，于当地输液治疗，为求进一步治疗，来我院就诊。\\n既往史: 无\\n个人史: 有新型冠状病毒疫苗接种史，共3针。\\n过敏史: \\n婚育史: \\n流行病史: \\n体格检查: T36.08℃。\\n辅助检查: \"}}

用户输入:
性别: 女\\n年龄: 45\\n主诉: 皮炎失眠高尿酸血症\\n现病史: 给予中药治疗。无发热、干咳、乏力、嗅（味）觉减退、鼻塞、流涕、咽痛、结膜炎、肌痛、腹泻\\n既往史: \\n个人史: 无发病前14天内有病例报告社区的旅行史或居住史；无发病前14天内与新型冠状病毒感染的患者或无症状感染者有接触史；无发病前14天内曾接触过来自有病例报告社区的发热或有呼吸道症状的患者；无聚集性发病；近14天内无进口冷链食品接触史。\\n过敏史: \\n婚育史: \\n流行病史: \\n体格检查: T36.5℃，P78次\\/分，R20次\\/分，BP142\\/78mmHg。心肺听诊未见异常，腹平软，无压痛。\\n辅助检查: 
你的输出:
{{\"reason\":\"1. 主诉中提到高尿酸血症，提示患者血尿酸水平升高；\\n2. 主诉中提到皮炎，提示存在皮肤炎症，可能为过敏性或接触性皮炎；\\n3. 主诉中提到失眠，提示存在睡眠障碍，可能与精神压力或其他疾病有关；\\n4. 体格检查中血压142\\/78mmHg，收缩压≥140mmHg，提示存在1级高血压；\\n5. 无发热、感染等症状，生命体征基本正常，心肺腹部检查未见明显异常。\",\"diseases\":\"1. 高尿酸血症；2. 皮炎；3. 失眠；4. 原发性高血压（1级）\",\"feature_content\":\"性别: 女\\n年龄: 45\\n主诉: 皮炎失眠高尿酸血症\\n现病史: 给予中药治疗。无发热、干咳、乏力、嗅（味）觉减退、鼻塞、流涕、咽痛、结膜炎、肌痛、腹泻\\n既往史: \\n个人史: 无发病前14天内有病例报告社区的旅行史或居住史；无发病前14天内与新型冠状病毒感染的患者或无症状感染者有接触史；无发病前14天内曾接触过来自有病例报告社区的发热或有呼吸道症状的患者；无聚集性发病；近14天内无进口冷链食品接触史。\\n过敏史: \\n婚育史: \\n流行病史: \\n体格检查: T36.5℃，P78次\\/分，R20次\\/分，BP142\\/78mmHg。心肺听诊未见异常，腹平软，无压痛。\\n辅助检查: \"}}

用户输入:
性别: 女\\n年龄: 25\\n主诉: 未避孕未怀孕3年\\n现病史: 2018年起与丈夫有性生活，2020年起未避孕未怀孕至今3年，2021年12月登记结婚。LMP2023.7.2，我科多次阴B提示PCO，诊断PCOS。\\n既往史: 无特殊\\n个人史: \\n过敏史: 无\\n婚育史: \\n流行病史: 暂无\\n体格检查: 血压（收缩压\\/舒张压）110\\/60mmHg  \\n  空腹血糖测定值：mmol\\/L  \\n  餐后血糖测定值：mmol\\/L  \\n  T：℃  \\n  P：次\\/分  \\n  R：次\\/分  \\n  神志：   \\n  呼吸：  \\n  \\n辅助检查: 2023-06-14，UU-DNA
你的输出:
{{\"reason\":\"1. 患者25岁，与丈夫未避孕未怀孕3年，符合不孕症的诊断标准；2. 患者多次经阴道B超提示多囊卵巢（PCO）表现，已诊断为多囊卵巢综合征（PCOS）；3. 末次月经为2023年7月2日，月经周期不规律，符合PCOS的临床特征；4. 体格检查血压正常，无特殊异常，辅助检查暂未见明显异常。\",\"diseases\":\"多囊卵巢综合征（PCOS）\",\"feature_content\":\"性别: 女\\n年龄: 25\\n主诉: 未避孕未怀孕3年\\n现病史: 2018年起与丈夫有性生活，2020年起未避孕未怀孕至今3年，2021年12月登记结婚。LMP2023.7.2，我科多次阴B提示PCO，诊断PCOS。\\n既往史: 无特殊\\n个人史: \\n过敏史: 无\\n婚育史: \\n流行病史: 暂无\\n体格检查: 血压（收缩压\\/舒张压）110\\/60mmHg  \\n  空腹血糖测定值：mmol\\/L  \\n  餐后血糖测定值：mmol\\/L  \\n  T：℃  \\n  P：次\\/分  \\n  R：次\\/分  \\n  神志：   \\n  呼吸：  \\n  \\n辅助检查: 2023-06-14，UU-DNA\"}}
"""


import json
from openai import OpenAI
deepseek = OpenAI(api_key="xxx", base_url="http://10.193.22.12:8080/v1")
cleaned_dataset_path = './datasets/cleaned_camp_data.jsonl'

# 被正确处理携带伪标签的数据
correct_with_label_cleaned_dataset_path = './datasets/correct_with_label_cleaned_camp_data.jsonl'
# 被错误处理携带伪标签的数据
err_with_label_cleaned_dataset_path = './datasets/err_with_label_cleaned_camp_data.jsonl'

def save_to_jsonl(file_path, data):
    """将数据写入JSONL文件"""
    with open(file_path, 'a', encoding='utf-8') as f:
        json_line = json.dumps(data, ensure_ascii=False)
        f.write(json_line + '\n')

# 借助DeepSeek生成训练数据伪标签
with open(cleaned_dataset_path, "r", encoding="utf-8") as file:
    # 逐行读取并解析 JSON 对象
    for line in file:
        # 将每一行解析为 JSON 对象
        cleaned_dataset_data = json.loads(line)
        prompt = gen_label_prompt.format(feature_content=cleaned_dataset_data.get("feature_content"))
        # 调用Deepseek 生成伪标签
        messages = [{'role': 'user', 'content': prompt}]
        try:
            response = deepseek.chat.completions.create(
                model="ycpc-gpt-r1",
                messages=messages
            )
            str_response = response.choices[0].message.content
            json_part = str_response.split("</think>")[-1].replace("```json", "").replace("```","")
            # 解析JSON
            try:
                parsed_json = json.loads(json_part)
                if isinstance(parsed_json, dict) and all(k in parsed_json for k in ['reason', 'diseases', 'feature_content']):
                    # 保存到正确文件
                    save_to_jsonl(correct_with_label_cleaned_dataset_path, parsed_json)
                    print("生成成功")
                else:
                    # JSON格式不正确，保存原始数据到错误文件
                    save_to_jsonl(err_with_label_cleaned_dataset_path, cleaned_dataset_data)
                    print("生成失败")
            except json.JSONDecodeError:
                # JSON解析失败，保存原始数据到错误文件
                save_to_jsonl(err_with_label_cleaned_dataset_path, cleaned_dataset_data)
                print("生成失败")
        except Exception as e:
            # API调用失败，保存原始数据到错误文件
            print(f"API调用失败，错误信息：{str(e)}")
            save_to_jsonl(err_with_label_cleaned_dataset_path, cleaned_dataset_data)