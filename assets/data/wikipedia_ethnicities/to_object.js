let data = []; // content of list_of_contemporary_ethics_groups.json

let keys = data[0];
data.shift();

data.map( valueList => {
let obj = valueList.reduce( (acum, curr, idx) => {  acum[keys[idx]] = curr; return acum; }  , {}); 
    console.log(obj);
return obj;
});
