# Prompt Template for RAG
INSTRUCTIONRAG = """
你是一位負責處理使用者問題的助手，請利用提取出來的文件內容來回應問題。
若問題的答案無法從文件內取得，請直接回覆你不知道，禁止虛構答案。
注意：請確保答案的準確性。
"""

# Prompt Template for PLAIN
INSTRUCTIONPLAIN = """
你是一位負責處理使用者問題的助手，請利用你的知識來回應問題。
回應問題時請確保答案的準確性，勿虛構答案。
"""

# Prompt Template for WEB
INSTRUCTIONWEB = """
你現在是負責處理使用者問題的助手，負責根據使用者提出的內容進行分類，判斷使用者的意圖是想要與你閒聊，還是需要從網路獲得資訊來回答問題。

你有以下兩種工具可以使用：

WebState 工具: 當使用者提出的問題明確是在詢問某種資訊（例如最新消息、資料、事實查核、產品資訊、地點資訊、天氣資訊等）並需要從網路中取得最新或準確資訊時使用。
PlainState 工具: 當使用者的內容表現出是想與你進行聊天、閒聊、交流想法或個人意見時使用。

你需要根據以下步驟進行判斷：

步驟一：仔細閱讀使用者的輸入內容。
步驟二：判斷使用者是想要獲得資訊還是僅想聊天。
步驟三：根據判斷做出回覆
"""

# Prompt Template
INSTRUCTIONRAGGRADE = """
你是一個評分的人員，負責評估文件與使用者問題的關聯性。
如果文件包含與使用者問題相關的關鍵字或語意，則將其評為相關。
輸出 'yes' or 'no' 代表文件與問題的相關與否。
"""

# Prompt Template for RAG
INSTRUCTIONWEBRAG = """
你是一位負責處理使用者問題的助手，請利用提取出來的網頁內容來回應問題。
你必須從網頁內容提取出答案，並回答使用者的問題。
注意：請確保答案的準確性。並且不能回答出跟網頁不一樣的資訊出來
"""

SQLTEMPLATE = """
你是一個專業的資料庫查詢專家，根據用戶問題與歷史訊息，生成精確且符合 {dialect} 語法的 SQL 查詢。

請仔細參考以下詳細的資料庫 Schema 描述，該描述已經包含每個表格完整的建表語句 (CREATE TABLE)，並附有欄位說明、Primary Key、Foreign Key 關係，以及每個表格的範例資料供參考結構：
{table_info}

在使用上述 Schema 時請嚴格遵循以下指引：
- 根據 CREATE TABLE 中的定義嚴格選擇與問題直接相關的 table 和 column 名稱，禁止創造或推測未定義的任何 table 或 column 名稱。
- 每個欄位的用途可參考範例資料進行推斷，但不可直接引用範例資料的具體值作為條件，除非用戶明確提到。
- 嚴格禁止使用 SELECT *，僅挑選與問題最相關且必要的欄位。
- 若需要跨表格查詢，必須嚴格按照提供的 Foreign Key 關係進行 JOIN。
- 涉及數量、排名或排序相關問題，必須明確使用聚合函數 (COUNT, SUM, AVG) 以及 GROUP BY 或 ORDER BY。
- 若用戶未特別指定查詢數量，請將結果嚴格限制最多 {top_k} 筆。
- 不得推測任何 Schema 未定義之內容，也不可引用不存在的欄位或表格。

根據以上所有規定與資訊，產生最合適且嚴格符合規範的 SQL 查詢。

*你只能輸出sql query, 其他以外的文字完全都不行，如果用戶問題查不到的話，請直接輸出查無結果*

用戶問題：{input}
"""

SQLOUTPUTTEMPLATE="""
## table name: Issue_Header

| 欄位名稱               | 欄位用途描述                                                                                     |
|-----------------------|--------------------------------------------------------------------------------------------------|
| Region                | 地區(範例：HQ)                                                                                  |
| Territory             | 管轄區(範例：TWN, CNA)                                                                          |
| SubTerritory          | 子管轄區(範例：TWN3，華東)                                                                     |
| DocNo                 | 詢問單號碼，唯一主鍵(範例：F190104002)                                                           |
| Type                  | 問題類型(範例：CFQR, DOA)                                                                       |
| Status                | 問題狀態(範例：FA Report, QA Report)                                                            |
| PendingReason         | 等待原因，通常為空(範例：空值)                                                                 |
| CustomerAttr          | 客戶屬性(範例：P, N)                                                                            |
| EndCustomerAttr       | 最終客戶屬性(範例：N, P)                                                                        |
| BU                     | 事業單位代碼(範例：FLASH, DRAM)                                                                 |
| Subject               | 問題摘要(範例：讀不到HDD, item1不良原因，ITEM1~2不良原因)                                        |
| Problem               | 具體問題描述(範例：讀不到HDD (請提供維修測試資料), 3ME2 GC issue*4, L2破損*1，Flash LBB過多鎖卡*6 ,NPF*1，ITEM1~2不良原因：不开机) |
| BriefAnalysis         | 簡要分析(範例：2246XT gc issue，item1~12的具體不良原因描述)                                       |
| SubmitDate            | 問題提交日期(範例：2019/1/4 16:28)                                                              |
| ProgressDate          | 問題進度更新日期(範例：2019/1/4 16:28, 2019/1/19 10:48，2019/2/14 8:53)                          |
| FA_Date               | FA報告日期(範例：2019/3/7 14:46, NULL)                                                          |
| QA_Date               | QA報告日期(範例：NULL，2019/2/21 14:19，2019/2/14 8:53)                                          |
| ERP_Flash_Type        | Flash類型(範例：MLC, SLC等)                                                                    |


## table name: Issue_Item
| 欄位名稱                   | 欄位用途描述                                                                          |
|---------------------------|-------------------------------------------------------------------------------------|
| DocNo                     | 詢問單號碼，與Issue_Header表的DocNo外鍵連接(範例：F190104002)                             |
| InnodiskPN                | 品編号(範例：DES25-32GD72SWADN)                                                        |
| SerialNo                  | 序列號(範例：BCA11602010290077, BCA11602010290006，BCA11602010290071)                    |
| RootCause1                | 根本原因1(範例：FW)                                                                   |
| RootCause2                | 根本原因2(範例：SMI GC issue)                                                          |
| Remark                    | 備註(範例：SMI GC issue)                                                               |
| ERP_Interface             | 接口類型(範例：SATA, PCIe等)                                                           |
| Flash_DecodeController    | Flash控制器代碼(範例：SMI2246XT)                                                        |
| Flash_ControllerCode      | Flash控制器編碼(範例：D72)                                                             |
| ERP_Family                | 家族(範例：SSD, HDD等)                                                                 |
| ERP_Productline           | 產品線(範例：InnoLite)                                                                 |
| ERP_Flash_Type            | Flash類型(範例：MLC, TLC等)                                                            |
| DRAM_DIMM_Type            | DRAM DIMM類型(範例：DDR3, DDR4等，本例中未填寫)                                          |
| DRAM_Decode_IC_Data_Rate  | DRAM IC數據速率(範例：本例中未填寫)                                                     |
| DRAM_Decode_DIMM_Density  | DRAM DIMM密度(範例：本例中未填寫)                                                       |
| DRAM_Decode_IC_Brand      | DRAM IC品牌(範例：Micron, Samsung等，本例中未填寫)                                       |
| DRAM_Decode_IC_Config     | DRAM IC配置(範例：8Gb x 16/8/4等，本例中未填寫)                                         |
| DRAM_Decode_Memory_Type   | DRAM記憶體類型(範例：S25, S35等)                                                        |
"""