# Dataset Schema (version 4.0.0)

Dataset schema provides a specification of an abstract dataset. It is contained in the [datasetSchema.json](../schemas/datasetSchema.json) file. An instance of this dataset schema is included with every dataset in the datasetDoc.json file. The semantics of datasetSchema.json and datasetDoc.json is [here](FAQ.md#semantics).  

Dataset schema specifies a dataset in three sections: about, dataResources, and qualities. Each of these sections are described below.

## About

The "about" section contains of some general information about the dataset and consists of the following fields.

| Field                 | Description                                                                                       | 
|-----------------------|---------------------------------------------------------------------------------------------------| 
| datasetID             | a unique ID assigned to a dataset                                                                 | 
| datasetName           | the name of a dataset                                                                             | 
| datasetURI            | the location of the dataset                                                                       | 
| description           | a brief description of the dataset                                                                | 
| citation              | citation of the source of the dataset                                                             | 
| humanSubjectsResearch | indicates if the dataset contains human subjects data or not                                      | 
| license               | license of the source of the dataset                                                              | 
| source                | source of the dataset                                                                             | 
| sourceURI             | location of the source of the dataset                                                             | 
| approximateSize       | size of the dataset                                                                               | 
| applicationDomain     | application domain of the dataset (e.g., medical, environment, transportation, agriculture, etc.) | 
| datasetVersion        | version of the current dataset                                                                    | 
| datasetSchemaVersion  | version of the datasetSchema                                                                      | 
| redacted              | a field indicating if the dataset has been redacted or not                                        |
| publicationDate       | publication date of the dataset, if available                                                     | 

## Data Resources

The "datasetResources" section annotates all the data resources in a dataset. A dataset is considered as a set of data resources. Therefore the "datasetResources" field is a list of dictionaries, one item per resource. Each dictionary item contains the following information about a data resource.

| Field                 | Description                                                                                       				| 
|-----------------------|-------------------------------------------------------------------------------------------------------------------| 
| resID                 | a unique ID assigned to a resource                                                                				| 
| resPath               | path of this resource relative to the dataset root                                            			        | 
| resType               | the type of this resource<sup>+</sup> 								                                                        | 
| resFormat             | a dictionary structure containing the occurring media types in this dataset as the keys and their corresponding list of file formats as the values <sup>++</sup>|
| isCollection          | a boolean flag indicating if this resource if a single item/file or a collection of items/files <sup>+++</sup>.    |

<sup>+</sup> __resType__ can be one of the following: __["image","video","audio","speech","text","graph","edgeList","table","timeseries","raw"]__

If `resType` is `edgeList`, the resource is basically a graph, but it is represented using an edge list. Each item in this list refers to an edge given by its source node and its target node (in that order, if it is a directed graph). For example, refer to [Case 5](#case-5) below. The edge list can also have optional edge attributes: [edgeID, source_node, target_node, edge_attr1, edge_attr2, ....]


<sup>++</sup> Here is an example of `resFormat`
```
"resFormat":{
	"image/jpeg":["jpeg", "jpg"],
	"image/png":["png"]
}
```

The [supportedResourceTypesFormats.json](supportedResourceTypesFormats.json) file contains the current list of all the resTypes, resFormats and the file types used in the datasets in a machine-readable way.


<sup>+++</sup>A **collection** of resources/files can be organized into directories and sub-directories.


If `resType` is "table" or "timeseries" (which is also a table basically), columns in the table can be annotated with the following information.
Not all columns are necessary annotated (see [minimal metadata datasets](./minimalMetadata.md) for an example).
There is an optional `columnsCount` field of the resource which can convey how many columns there are,
which can be used to determine if all columns are annotated.

| Field                 | Description                                                                  	| 
|-----------------------|-------------------------------------------------------------------------------| 
| colIndex              | index of the column                                                          	| 
| colName               | name of the column                                                		    | 
| colDescription        | description of a column                                                       |
| colType               | the data type of this column                                                  | 
| role                  | the role this column plays                                                    | 
| refersTo              | an optional reference to another resource if applicable                   	|
| timeGranularity       | if a particular column represents time axis, this field captures the time sampling granularity as a numeric `value` and `unit` (e.g., 15-second granularity). |

A column's type can be __one of__ the following:

| Type                 | Description                                                                   | 
|----------------------|-------------------------------------------------------------------------------| 
| boolean     ||
| integer     ||
| real        ||
| string      ||
| categorical ||
| dateTime    ||
| realVector  | This is stored as a string in CSV files. It's represented as "val1,val2,vl3,val4,...". For example, in the objectDetection task, the bounding boundary is a rectangle, which is represented as: "0,160,156,182,276,302,18,431". |
| json        ||
| geojson     | See here: http://geojson.org/ |
| unknown     | Column type is unknown. |


A column's role can be __one or more__ of the following:

| Role                    | Description 	| 
|-------------------------|-----------------| 
| index                   | Primary index of the table. Should have unique values without missing values | 
| multiIndex              | Primary index of the table. Values in this primary index are not necessary unique to allow the same row to be repeated multiple times (useful for certain task types like multiLabel and objectDetection) |
| key                     | Any column that satisfies the uniqueness constraint can splay the role of a key. A table can have many keys. Values can be missing in such column. A key can be referenced in other tables (foreign key)|
| attribute               | The column contains an input feature or attribute that should be used for analysis | 
| suggestedTarget         | The column is a potential target variable for a problem. If a problem chooses not to employ it as a target, by default, it will also have an attribute role  | 
| timeIndicator           | Entries in this column are time entries | 
| locationIndicator       | Entries in this column correspond to physical locations  |
| boundaryIndicator       | Entries in this column indicated boundary (e.g., start/stop in audio files, bounding-boxes in an image files, etc.) |
| interval                | Values in this column represents an interval and will contain a "realVector" of two values|
| instanceWeight          | This is a weight assigned to the row during training and testing. It is used when a single row or pattern should be weighted more or less than others.|
| boundingPolygon         | This is a type of boundary indicator representing geometric shapes. It's type is "realVector". It contains two coordinates for each polygon vertex - always even number of elements in string. The vertices are ordered counter-clockwise. For e.g., a bounding box in objectDetection consisting of 4 vertices and 8 coordinates: "0,160,156,182,276,302,18,431"  |
| suggestedPrivilegedData | The column is potentially unavailable variable during testing for a problem. If a problem chooses not to employ it as a privileged data, by default, it will also have an attribute role  | 
| suggestedGroupingKey    | A column with this role is a potential candidate for group_by operation (rows having the same values in this column should be grouped togehter). Multiple columns with this role can be grouped to obtain a hierarchical multi-index data. |
| edgeSource | Entries in this column are source nodes of edges in a graph |
| directedEdgeSource | Entries in this column are source nodes of directed edges in a graph |
| undirectedEdgeSource | Entries in this column are source nodes of undirected edges in a graph |
| multiEdgeSource | Entries in this column are source nodes of edges in a multi graph |
| simpleEdgeSource | Entries in this column are source nodes of edges in a simple graph |
| edgeTarget | Entries in this column are target nodes of edges in a graph |
| directedEdgeTarget | Entries in this column are target nodes of directed edges in a graph |
| undirectedEdgeTarget | Entries in this column are target nodes of undirected edges in a graph | 
| multiEdgeTarget | Entries in this column are target nodes of edges in a multi graph |
| simpleEdgeTarget | Entries in this column are target nodes of edges in a simple graph |

A time granularity unit can be __one of__ the following:

| Unit                 | Description                                                                   | 
|----------------------|-------------------------------------------------------------------------------| 
| seconds     ||
| minutes     ||
| days        ||
| weeks       ||
| months      ||
| years       ||
| unspecified ||

__Notes regarding index, multi-index roles__
A table can have only one `index` or `multiIndex` column.

For a table with `multiIndex` column, for a particular index value there can be multiple rows. Those rows have exactly the same values in all columns except columns with `suggestedTarget` role.

Of special note is the "refersTo" field, which allows linking of tables to other entities. A typical "refersTo" entry looks like the following.
```
"colIndex":1,
  "colName":"customerID",
  "colType":"integer",
  "role":["attribute"],
  "refersTo":{
    "resID":"0",
    "resObject":{
      "columnName":"custID"}}
```
Here, the customerID column in one table is linking to another resource whose "resID"="0", which is presumably a different table. Further, the object of reference, "resObject", is a column whose "columnName"="custID".

A table entry can also refer to other kinds of resources besides another tables, including, a raw file in a collection of raw files (e.g., img1.png in a collection of images), a node in a graph, an edge in a graph. The below table lists the different resource objects that can be referenced. 

| resObject | resource referenced | Description |
|-----------|---------------------|-------------|
| columnIndex, columnName | table | Entries in this column refer to entries in another table (foreign key). See [example](#case-3) |
| item | a collection of raw files | Each entry in this column refers to an item in a collection of raw files. See [example](#case-2) |
| nodeAttribute | graph | Entries in this column refer to attribute values of a node in a graph. This can include a reference to the nodeID|
| edgeAttribute | graph | Entries in this column refer to attribute values of an edge in a graph. This can include a reference to the edgeID|

__Notes regarding graph node/edge roles__
The parent roles of egdeSource/edgeTarget can be combined with derived roles such as directedEdgeSource/directedEdgeTarget and undirectedEdgeSource/undirectedEdgeTarget to indicate directed or undirected edges in a graph. Similarly, the metadata can capture information about simple/multi-edges using multiEdgeSource/multiEdgeTarget and simpleEdgeSource/simpleEdgeTarget.


Next, we will see how this approach of treating everything as a resource and using references to link resources can handle a range of cases that can arise.

## Case 1: Single table

In many openml and other tabular cases, all the learning data is contained in a single tabular file. In this case, an example dataset will look like the following.
```
<dataset_id>/
|-- tables/
	|-- learningData.csv
		d3mIndex,sepalLength,sepalWidth,petalLength,petalWidth,species
		0,5.2,3.5,1.4,0.2,I.setosa
		1,4.9,3.0,1.4,0.2,I.setosa
		2,4.7,3.2,1.3,0.2,I.setosa
		3,4.6,3.1,1.5,0.2,I.setosa
		4,5.0,3.6,1.4,0.3,I.setosa
		5,5.4,3.5,1.7,0.4,I.setosa
		...
|-- datasetDoc.json
```
The [datasetDoc.json](examples/iris.datasetDoc.json) for this example shows how this case is handled. Note that in this datasetDoc.json, the "role" of the column "species" is "suggestedTarget". The reason for this is given [here](FAQ.md#suggested-targets).

## Case 2: Single table, multiple raw files <a name="case-2"></a>

Consider the following sample image classification dataset. It has one learningData.csv file, whose "image" column has pointers to image files.
```
<dataset_id>/
|-- media/
	|-- img1.png
	|-- img2.png
	|-- img3.png
	|-- img4.png
|-- tables/
	|-- learningData.csv
		d3mIndex,image,label
		0,img1.png,cat
		1,img2.png,dog
		2,img3.png,dog
		3,img4.png,cat
|-- datasetDoc.json
```
The [datasetDoc.json](examples/image.datasetDoc.json) for this example shows how this case is handled. Two things to note in this datasetDoc.json are:
- We define all the images in media/ as a single resource, albeit a collective resource (notice "isCollection"=true).
```
{
  "resID": "0",
  "resPath": "media/",
  "resType": "image",
  "resFormat": ["img/png"],
  "isCollection": true, 
}
```
- We reference this image-collection resource ("resID"="1") in the "image" column of the "learningData.csv" resource:
```
{
  "colIndex": 1,
  "colName": "image",
  "colType": "string",
  "role": "attribute",
  "refersTo":{
    "resID": "0",
    "resObject": "item"
  }
}
```
The semantics here is as follows: The entries in the "image" column refers to an item ("resObject"="item") in a resource whose "resID"="0". Therefore, an entry 'img1.png' in this column refers to an item of the same name in the image collection. Since we know the relative path of this image-collection resource ("resPath"="media/"), we can locate "img1.png" at <dataset_id>/media/img1.png.

## Case 3: Multiple tables referencing each other <a name="case-3"></a>
Consider the following dataset containing multiple relational tables referencing each other.
```
<dataset_id>/
|-- tables/
	|-- customers.csv
		custID,country,first_invoices_time,facebookHandle
		0,USA,12161998,"johnDoe"
		1,USA,06272014,"janeSmith"
		2,USA,12042014,"fooBar"
		3,AUS,02022006,"johnyAppleseed"
		...
	|-- items.csv
		stockCode,first_item_purchases_time,Description
		FIL36,02162005,waterfilter
		MAN42,06272014,userManual
		GHDWR2,01112012,generalHardware
		...
	|-- invoices.csv
		invoiceNo,customerID,first_item_purchases_time
		0,734,12161998
		1,474,11222010
		2,647,10182011
		...
	|-- learningData.csv
		item_purchase_ID,invoiceNo,invoiceDate,stockCode,unitPrice,quantity,customerSatisfied,profitMargin
		0,36,03142009,GYXR15,35.99,2,1,0.1,
		1,156,02022006,TEYT746,141.36,2,0,0.36
		2,8383,11162010,IUYU132,57.25,2,0,0.22
		3,3663,12042014,FURY145,1338.00,1,1,0.11
		4,6625,07072013,DSDV97,762.00,1,1,0.05
		...
```
Here, "learningData" references both "invoice" and "items" tables. "invoice" table in turn references "customers" table. The [datasetDoc.json](examples/multitable.datasetDoc.json) for this example shows how this case is handled. 

There are a number of important things to notice in this example:

- The "isColleciton" flag is False for all the resources, indicating that there are no collection-of-files resource as was the case in the image example above.

- As shown in  the snippet below, columns in one resource are referencing columns in another resource.
```
  -- In "resID"="1" ---        -- In "resID"="3" ---      -- In "resID"="3" ---
  "colIndex":1,                "colIndex":1,              "colIndex":3,
  "colName":"customerID",      "colName":"invoiceNo",     "colName":"stockCode",
  "colType":"integer",         "colType":"integer",       "colType":"string",
  "role":["attribute"],        "role":["attribute"],      "role":["attribute"],
  "refersTo":{                 "refersTo":{               "refersTo":{
    "resID":"0",                 "resID":"1",               "resID":"2",
    "resObject":{                "resObject":{              "resObject":{
      "columnName":"custID"}}      "columnIndex":0}}          "columnName":"stockCode"}}
``` 
- A column can have multiple roles. For instance in "customer.csv", the column "facebookHandle" has two roles ("role":["attribute", "key"]). It is a key because it satisfies the uniqueness constraint. (see [here](FAQ.md#index-key) for distinction between index and key roles.)
```
{
  "colIndex":3,
  "colName":"facebookHandle",
  "colType":"string",
  "role":["attribute", "key"]
}
```
- Finally, a column can be referenced using its columnIndex or columnName. See "resID"=3 entries above. Ideally we want column names participating in a reference for readability reasons, however, occasionally we have come across datasets with tables that do not have unique column names. In such cases, columnIndex is used.

## Case 4: Referencing Graph Elements <a name="case-4"></a>
Assume a sample graph resource below, G2.gml, whose 'resID' = 1
```
graph [
  node [
    id 0
    label "1"
    nodeID 1
  ]
  node [
    id 1
    label "88160"
    nodeID 88160
  ]
...
```

A corresponding learningData.csv

| d3mIndex | G2.nodeID | class |
|----------|-----------|-------|
|0         |88160      |1      |
|...       |...        |...    |


A corresponding datasetDoc.json
```
{
    "colIndex": 2,
    "colName": "G2.nodeID",
    "colType": "integer",
    "role": [
        "attribute"
    ],
    "refersTo": {
    "resID": "1",
    "resObject": "node"
    }
},
```

- "resID": "1" - this points to the graph resource G2.gml
- "resObject": "node" - this points to a node in that graph. **All nodes have a 'nodeID' in the GML file and all nodes are uniquely identified and referenced by its nodeID.**
- So, in the learningData table above, row 0 points to a node in G2.gml whose nodeID = 88160:

```
  node [
    id 1
    label "88160"
    nodeID 88160
  ]
```

**Note:** nodeID's can also be strings as shown in the example below:
```
graph.gml (resID=1)                      learningData.csv                                      datasetDoc.json
===================                      =================                                     ================
graph [                                  | d3mIndex | child.nodeID | parent.nodeID | bond |    ...
  node [                                 |----------|--------------|---------------|------|    {
    id 0                                 | 0        | BSP          | OPSIN         | C    |      "colIndex": 1,
    label "Protein Acyl Transferases"    | 1        | RHOD         | OPSIN         | NC   |      "colName": "child.nodeID",
    nodeID "PATS"                        | 2        | RXN          | GPCR          | NC   |      "colType": "string",
  ]                                      | ...      | ...          | .....         | ...  |      "role": [
  node [                                                                                            "attribute"
    id 1                                                                                         ],
    label "Blue Sensitive Opsins"                                                                "refersTo": {
    nodeID "BSoP"                                                                                   "resID": "1",
  ]                                                                                                 "resObject": "node"
...                                                                                              }},
                                                                                                ...
```

## Case 5: Graph resources represented as edge lists <a name="case-5"></a>

In v3.1.1, we introduced a new "resType" called "edgeList". This is how a dataset migth look for a case where the graphs are represented using edge lists.

```
<dataset_id>/
  |-- tables/
    |-- learningData.csv
      d3mIndex,user_ID,user_location,gender,age,high_valued_user
      0,2048151474,46614,M,37,"johnDoe",1
      1,6451671537,08105,F,55,"janeSmith",1
      2,6445265804,31021,F,23,"fooBar",1
      3,0789614390,11554,58,"johnyAppleseed",1
      ...
    |-- graphs
      |-- mentions.edgelist.csv
        edgeID,user_ID,mentioned_userID
        0,2048151474,0688363262
        1,2048151474,1184169266
        2,2048151474,3949024994
        3,2048151474,7156662506
        ...
```

It's corresponding datasetDoc.json will look like the following:
```
  ...
  {
    "resID": "0",
    "resPath": "graphs/mentions.edgelist.csv",
    "resType": "edgeList",
    "resFormat": ["text/csv"],
    "isCollection": false,
    "columns": [
      {
        "colIndex": 0,
        "colName": "edgeID",
        "colType": "integer",
        "role": [
            "index"
        ],
      },
      {
        "colIndex": 1,
        "colName": "user_ID",
        "colType": "integer",
        "role": [
            "attribute"
        ],
        "refersTo": {
          "resID": "1",
          "resObject":{
            "columnName":"user_ID"}
          }
      },
      {
        "colIndex": 2,
        "colName": "mentioned_userID",
        "colType": "integer",
        "role": [
            "attribute"
        ],
        "refersTo": {
          "resID": "1",
          "resObject":{
            "columnName":"user_ID"}
          }
      }
    ]
  },
  {
    "resID": "learningData",
    "resPath": "tables/learningData.csv",
    "resType": "table",
    "resFormat": ["text/csv"],
    "isCollection": false,
    "columns": [
      {
        "colIndex": 0,
        "colName": "d3mIndex",
        "colType": "integer",
        "role": [
            "index"
        ],
      },
      {
        "colIndex": 1,
        "colName": "user_ID",
        "colType": "integer",
        "role": [
            "attribute"
        ]
      },
      {
        "colIndex": 2,
        "colName": "user_location",
        "colType": "categorical",
        "role": [
            "attribute"
        ],
      },
    ...
  },
...
```

# Qualities

A dataset can be annotated with its unique properties or __qualities__. The "qualities" section of the datasetSchema specifies the fields that can be used to used to capture those qualities.

| Field | Description |
|-------|-------------|
| qualName | name of the quality variable |
| qualValue | value of the quality variable |
| qualValueType | data type of the quality variable; can be one of ["boolean","integer","real","string","dict"]|
| qualValueUnits | units of the quality variable |
| restrictedTo | if a quality is restricted to specific resource and not the entire dataset, this field captures that constraint |

__Note:__ We are not providing a taxonomy of qualities but, through this section in the dataset schema, are providing a provision for capturing the qualities of a dataset in its datasetDoc.json file.

Some qualities apply to an entire dataset. For instance, if one wants to annotate a dataset with (1) information about the presence of multi relational tables or not, (2) requires LUPI (Learning Using Privileged Information) or not, and/or (3) its approximate size:
```
"qualities":[
  {
    "qualName":"multiRelationalTables",
    "qualValue":true,
    "qualValueType":"boolean"
  },
  {
    "qualityName":"LUPI",
    "qualityValue":true,
    "qualValueType":"boolean"
  },
  {
    "qualName":"approximateSize",
    "qualValue":300,
    "qualValueType":"integer"
    "qualValueUnits":"MB"
}]
```

Some qualities are applicable only to certain resources within a dataset. For instance, "maxSkewOfNumericAttributes" is table specific and not dataset wide. In such cases, we include the "restrictedTo" field:
```
{
  "qualName":"maxSkewOfNumericAttributes",
  "qualValue":6.57,
  "qualValueType":"real",
  "restrictedTo":{
    "resID":"3"			   // redID 3 is a table
 }
},
```

Further, some qualities are restricted to not only a specific resource but to a specific component within a resource. For instance, "classEntropy" and "numMissingValues" are only applicable to a column within a table within a dataset. In such cases, the "restrictedTo" field will have additional "resComponent" qualifier:
```
{
  "qualName":"classEntropy",
  "qualValue":3.63
  "qualValueType":"real",
  "restrictedTo":{
    "resID":"3",
    "resComponent":{"columnName":"customerSatisfied"}
  }
},
{
  "qualName":"numMissingValues",
  "qualValue":12,
  "qualValueType":"integer",
  "restrictedTo":{
    "resID":"3",
    "resComponent":{"columnName":"customerSatisfied"}
  }
}
```
Similarly a "privilegedFeature" called "facebookHandle" will be restricted to its column within a table (resID=0, customer table) within a dataset as shown below:
```
{
  "qualName":"privilegedFeature",
  "qualValue":true,
  "qualValueType":"boolean",
  "restrictedTo":{
    "resID":"0",
    "resComponent":{"colName":"facebookHandle"}
  }
}
```

For a full example see the [datasetDoc.json](examples/multitable.datasetDoc.json) file for the above Case 3.

"resComponent" can also be strings "nodes" and "edges".

# FAQ

#### Why do we have distinct "index" and "key" roles

An field which has unique values is a key. However, a key that is used to uniquely identify a row in a table is an index. There can be multiple keys in a table, but a single index. Both index and key columns can participate in the foreign key relationship with other tables.

#### Why is "role" filed a list?

A column can play multiple roles. For instance, it can be a "locationIndicator" and an "attribute", a "key" and an "attribute", and so on. 

#### How are LUPI datasets represented

LUPI datasets contain columns for variables which are not available during testing (testing dataset split does not contain data for those columns).

Such dataset can have some columns marked with role "suggestedPrivilegedData" to suggest that those columns might be used in the problem description as LUPI columns. The problem description has a list of such columns specified through "privilegedData" list of columns.

#### Why are we using "suggestedTarget" role instead of "target" in datasetSchema?

Defining targets is inherently in the domain of a problem. The best we can do in a dataset is to make a suggestion to the problem that these are potential targets. Therefore we use "target" in the problemSchema vocabulary and "suggestedTarget" in the datasetSchema vocabulary.

#### What is the use for the role 'boundaryIndicator'
In some datasets, there are columns that denote boundaries and is best not treated as attributes for learning. In such cases, this tag comes in handy. Example:

```
d3mIndex,audio_file,startTime,endTime,event
0,a_001.wav,0.32,0.56,car_horn
1,a_002.wav,0.11,1.34,engine_start
2,a_003.wav,0.23,1.44,dog_bark
3,a_004.wav,0.34,2.56,dog_bark
4,a_005.wav,0.56,1.22,car_horn
...
```
Here, 'startTime' and 'endTime' indicate the start and end of an event of interest in the audio file. These columns are good candidates for the role of boundary indicators.

A boundary column can contain `refersTo` reference to point to the column for which it is a boundary.
If `refersTo` is not provided it means that a boundary column is a boundary for the first preceding non-boundary column.
E.g., in the example above, `startTime` and `endTime` columns are boundary columns for `audio_file` column
(or more precisely, for files pointed to by the `audio_file` column).
