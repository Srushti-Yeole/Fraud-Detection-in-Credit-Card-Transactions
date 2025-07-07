form = document.getElementById("form")
form.addEventListener('submit', (e)=>{
    isEmpty = false
    inputs = [...Array(28).keys().map((i)=>`V${i+1}`)]
    inputs.push("amount")
    inputs.forEach((i)=>{
        element = document.getElementById(i)
        if(element.value === "" || element.value == null){
            element.setAttribute("aria-invalid", "true")
            isEmpty = true
        }
        else{
            element.setAttribute("aria-invalid", "false")
        }
    })
    
    if(isEmpty){
        alert("Please fill all the fields")
        e.preventDefault();
    }
})

function redirect(){
    window.history.go(-1)
}