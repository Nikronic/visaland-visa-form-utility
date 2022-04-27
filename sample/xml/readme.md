# How to read XML files
This readme contains the keywords, variables, IDs, etc used to describe different fields in the XFA format used by Adobe in PDF files

## General Warning
1. The VSCode extension I used to view and format the XML code has bug and removed the trailing ">" parts for all tags. Also, it added "=" for no reason to start of tags.
2. 

## 5257e-9.xml
- starting line: 1215
- ServiceIn: 01 = Eng
- Country: 223 = Iran, 045 = Turkey
  - PlaceBirthCountry: 223 = Iran
  - Citizenship: 223 = Iran
  - Status: 01 = Citizenship, 06 = Other
- PCRIndicator: N = No, Previous country of residence
- CurrentCOR: Current country of Residence
- MaritalStatus: 01 = Married
- nativeLang: 223 = Farsi
- PrevMarriedIndicator: have you had previous marriage
- natIDIndicator: national ID indicator
- usCardIndicator: US green card holder
- Phone Type: 02 = Other
- PurposeOfVisit: 08 = Family
- Funds: some number in USD or CAD
- EducationIndicator: have you had secondary education?

## 5645e-17
Warnings:
1. SectionC which contains information about BROTHERS AND SISTERS uses the same keys as SectionB, i.e. Brother or Sister are tagged as Child! Alabama simulator, KKonaW.
2. TBD


- Starting line: 1
- Subform1: Type of visit -> Visitor, Worker, Student, Other
- Applicant
  - ChildMStatus: 5 = Married / Physically Present
- Spouse
  - SpouseYes: 0 = No, Spouse is coming
  - SpouseNo: 1 = Yes, Spouse is NOT coming
- Child
  - ChildYes: 0 = No, Child is coming
  - ChildNo: 1 = Yes, Child is NOT coming

