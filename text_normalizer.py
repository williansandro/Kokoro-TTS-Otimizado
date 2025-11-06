import re
from datetime import datetime
from typing import Dict, List, Tuple

class KokoroTextNormalizer:
    """
    Normaliza texto convertendo números, datas, símbolos e operações matemáticas
    para forma escrita, preservando toda a informação original.
    """
    
    # Mapeamentos básicos
    UNIDADES = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }
    
    DEZENAS = {
        '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
        '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
        '18': 'eighteen', '19': 'nineteen', '20': 'twenty', '30': 'thirty',
        '40': 'forty', '50': 'fifty', '60': 'sixty', '70': 'seventy',
        '80': 'eighty', '90': 'ninety'
    }
    
    MESES_EN = {
        '01': 'January', '02': 'February', '03': 'March', '04': 'April',
        '05': 'May', '06': 'June', '07': 'July', '08': 'August',
        '09': 'September', '10': 'October', '11': 'November', '12': 'December'
    }
    
    ORDINAIS = {
        '1': 'first', '2': 'second', '3': 'third', '4': 'fourth', '5': 'fifth',
        '6': 'sixth', '7': 'seventh', '8': 'eighth', '9': 'ninth', '10': 'tenth',
        '11': 'eleventh', '12': 'twelfth', '13': 'thirteenth', '20': 'twentieth',
        '21': 'twenty-first', '22': 'twenty-second', '23': 'twenty-third',
        '30': 'thirtieth', '40': 'fortieth', '50': 'fiftieth', '100': 'one hundredth'
    }
    
    ROMANOS = {
        'I': 'the First', 'II': 'the Second', 'III': 'the Third', 'IV': 'the Fourth',
        'V': 'the Fifth', 'VI': 'the Sixth', 'VII': 'the Seventh', 'VIII': 'the Eighth',
        'IX': 'the Ninth', 'X': 'the Tenth', 'XI': 'the Eleventh', 'XII': 'the Twelfth',
        'XIV': 'the Fourteenth', 'XV': 'the Fifteenth', 'XVI': 'the Sixteenth'
    }
    
    @staticmethod
    def numero_para_palavras(num: int) -> str:
        """Converte número inteiro para palavras em inglês."""
        if num == 0:
            return 'zero'
        
        if num < 0:
            return 'negative ' + KokoroTextNormalizer.numero_para_palavras(-num)
        
        if num < 10:
            return KokoroTextNormalizer.UNIDADES[str(num)]
        
        if num < 20:
            return KokoroTextNormalizer.DEZENAS[str(num)]
        
        if num < 100:
            dezenas = (num // 10) * 10
            unidade = num % 10
            if unidade == 0:
                return KokoroTextNormalizer.DEZENAS[str(dezenas)]
            return f"{KokoroTextNormalizer.DEZENAS[str(dezenas)]}-{KokoroTextNormalizer.UNIDADES[str(unidade)]}"
        
        if num < 1000:
            centenas = num // 100
            resto = num % 100
            resultado = f"{KokoroTextNormalizer.UNIDADES[str(centenas)]} hundred"
            if resto > 0:
                resultado += f" and {KokoroTextNormalizer.numero_para_palavras(resto)}"
            return resultado
        
        if num < 1000000:
            milhares = num // 1000
            resto = num % 1000
            resultado = f"{KokoroTextNormalizer.numero_para_palavras(milhares)} thousand"
            if resto > 0:
                resultado += f" {KokoroTextNormalizer.numero_para_palavras(resto)}"
            return resultado
        
        if num < 1000000000:
            milhoes = num // 1000000
            resto = num % 1000000
            resultado = f"{KokoroTextNormalizer.numero_para_palavras(milhoes)} million"
            if resto > 0:
                resultado += f" {KokoroTextNormalizer.numero_para_palavras(resto)}"
            return resultado
        
        return str(num)
    
    @staticmethod
    def numero_ordinal(num: int) -> str:
        """Converte número para ordinal em inglês."""
        if num in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            return KokoroTextNormalizer.ORDINAIS[str(num)]
        
        if num < 20:
            return f"{KokoroTextNormalizer.numero_para_palavras(num)}th"
        
        if 20 <= num < 100:
            dezenas = (num // 10) * 10
            unidade = num % 10
            if unidade == 0:
                return f"{KokoroTextNormalizer.ORDINAIS.get(str(dezenas), f'{dezenas}th')}"
            unidade_ord = KokoroTextNormalizer.ORDINAIS.get(str(unidade), f"{unidade}th")
            dezena_nome = KokoroTextNormalizer.DEZENAS[str(dezenas)].replace('y', 'ieth')
            return f"{dezena_nome.replace('ty', 'tieth')}-{unidade_ord}"
        
        return f"{KokoroTextNormalizer.numero_para_palavras(num)}th"
    
    @staticmethod
    def processar_numeros(texto: str) -> str:
        """Processa números ordinais, cardinais e especiais."""
        
        # Ordinais simples (1st, 2nd, etc)
        def converter_ordinal(match):
            num = int(match.group(1))
            return KokoroTextNormalizer.numero_ordinal(num)
        
        texto = re.sub(r'(\d+)(?:st|nd|rd|th)\b', converter_ordinal, texto)
        
        # Números com separadores (1,000 ou 1.000)
        def converter_numero_separado(match):
            num_str = match.group(0).replace(',', '').replace('.', '')
            if num_str.isdigit():
                return KokoroTextNormalizer.numero_para_palavras(int(num_str))
            return match.group(0)
        
        texto = re.sub(r'\d{1,3}(?:[,\.]\d{3})+(?!\d)', converter_numero_separado, texto)
        
        # Números simples (2 dígitos ou mais, sem vírgulas)
        def converter_numero_simples(match):
            num = int(match.group(0))
            if num > 9:  # Apenas números maiores que 9
                return KokoroTextNormalizer.numero_para_palavras(num)
            return match.group(0)
        
        texto = re.sub(r'\b([1-9][0-9]+)\b', converter_numero_simples, texto)
        
        return texto
    
    @staticmethod
    def processar_datas(texto: str) -> str:
        """Processa datas em diferentes formatos."""
        
        # Formato: 05/03/1842 ou 05-03-1842
        def converter_data_numerica(match):
            partes = re.split(r'[/-]', match.group(0))
            if len(partes) == 3:
                mes = partes[0]
                dia = partes[1]
                ano = partes[2]
                
                mes_nome = KokoroTextNormalizer.MESES_EN.get(mes, mes)
                dia_num = KokoroTextNormalizer.numero_para_palavras(int(dia))
                ano_num = KokoroTextNormalizer.numero_para_palavras(int(ano))
                
                return f"{mes_nome} {dia_num}, {ano_num}"
            return match.group(0)
        
        texto = re.sub(r'\d{2}[/-]\d{2}[/-]\d{4}', converter_data_numerica, texto)
        
        return texto
    
    @staticmethod
    def processar_percentuais(texto: str) -> str:
        """Processa percentuais."""
        
        def converter_percentual(match):
            numero = match.group(1).replace(',', '.')
            try:
                num_float = float(numero)
                num_palavras = KokoroTextNormalizer.numero_para_palavras(int(num_float))
                
                # Se tem decimal
                if '.' in numero:
                    partes = numero.split('.')
                    if len(partes[1]) == 1:
                        decimal_palavras = KokoroTextNormalizer.numero_para_palavras(int(partes[1]))
                        return f"{num_palavras} point {decimal_palavras} percent"
                    else:
                        decimal_completo = KokoroTextNormalizer.numero_para_palavras(int(partes[1]))
                        return f"{num_palavras} point {decimal_completo} percent"
                
                return f"{num_palavras} percent"
            except:
                return match.group(0)
        
        texto = re.sub(r'([\d,]+(?:\.\d+)?)\s*%', converter_percentual, texto)
        
        return texto
    
    @staticmethod
    def processar_operacoes_matematicas(texto: str) -> str:
        """Processa operações matemáticas básicas."""
        
        mapeamentos = {
            r'(\d+)\s*\+\s*(\d+)': lambda m: f"{KokoroTextNormalizer.numero_para_palavras(int(m.group(1)))} plus {KokoroTextNormalizer.numero_para_palavras(int(m.group(2)))}",
            r'(\d+)\s*[-–]\s*(\d+)': lambda m: f"{KokoroTextNormalizer.numero_para_palavras(int(m.group(1)))} minus {KokoroTextNormalizer.numero_para_palavras(int(m.group(2)))}",
            r'(\d+)\s*[×x*]\s*(\d+)': lambda m: f"{KokoroTextNormalizer.numero_para_palavras(int(m.group(1)))} times {KokoroTextNormalizer.numero_para_palavras(int(m.group(2)))}",
            r'(\d+)\s*[÷/]\s*(\d+)': lambda m: f"{KokoroTextNormalizer.numero_para_palavras(int(m.group(1)))} divided by {KokoroTextNormalizer.numero_para_palavras(int(m.group(2)))}",
        }
        
        for padrao, funcao in mapeamentos.items():
            texto = re.sub(padrao, funcao, texto, flags=re.IGNORECASE)
        
        return texto
    
    @staticmethod
    def processar_fracoes(texto: str) -> str:
        """Processa frações."""
        
        fracao_map = {
            r'1/2': 'one-half',
            r'1/3': 'one-third',
            r'2/3': 'two-thirds',
            r'3/4': 'three-quarters',
            r'1/4': 'one-quarter',
        }
        
        for fracao, palavras in fracao_map.items():
            texto = texto.replace(fracao, palavras)
        
        return texto
    
    @staticmethod
    def processar_decimais(texto: str) -> str:
        """Processa números decimais."""
        
        def converter_decimal(match):
            numero = match.group(0)
            partes = numero.split('.')
            
            inteira = KokoroTextNormalizer.numero_para_palavras(int(partes[0])) if partes[0] else 'zero'
            decimal_str = partes[1]
            
            # Ler cada dígito do decimal
            decimal_lido = ' '.join([KokoroTextNormalizer.UNIDADES[d] for d in decimal_str])
            
            return f"{inteira} point {decimal_lido}"
        
        texto = re.sub(r'\d+\.\d+', converter_decimal, texto)
        
        return texto
    
    @staticmethod
    def processar_unidades_medida(texto: str) -> str:
        """Processa unidades de medida."""
        
        mapeamentos = {
            r'(\d+)\s*km\b': lambda m: f"{KokoroTextNormalizer.numero_para_palavras(int(m.group(1)))} kilometers",
            r'(\d+)\s*kg\b': lambda m: f"{KokoroTextNormalizer.numero_para_palavras(int(m.group(1)))} kilograms",
            r'(\d+)\s*m\b(?!m)': lambda m: f"{KokoroTextNormalizer.numero_para_palavras(int(m.group(1)))} meters",
            r'(\d+)\s*L\b': lambda m: f"{KokoroTextNormalizer.numero_para_palavras(int(m.group(1)))} liters",
            r'(\d+)\s*ml\b': lambda m: f"{KokoroTextNormalizer.numero_para_palavras(int(m.group(1)))} milliliters",
            r'(\d+)\s*°C': lambda m: f"{KokoroTextNormalizer.numero_para_palavras(int(m.group(1)))} degrees Celsius",
        }
        
        for padrao, funcao in mapeamentos.items():
            texto = re.sub(padrao, funcao, texto, flags=re.IGNORECASE)
        
        return texto
    
    @staticmethod
    def processar_valores_monetarios(texto: str) -> str:
        """Processa valores monetários."""
        
        # R$ 50 -> fifty reais
        texto = re.sub(r'R\$\s*([\d,]+(?:\.\d{2})?)', 
                      lambda m: f"{KokoroTextNormalizer.numero_para_palavras(int(m.group(1).replace(',', '').split('.')[0]))} reais",
                      texto)
        
        # $ 10.50 -> ten dollars and fifty cents
        texto = re.sub(r'\$\s*([\d,]+(?:\.\d{2})?)',
                      lambda m: f"{KokoroTextNormalizer.numero_para_palavras(int(m.group(1).split('.')[0]))} dollars",
                      texto)
        
        # € 100 -> one hundred euros
        texto = re.sub(r'€\s*([\d,]+(?:\.\d{2})?)',
                      lambda m: f"{KokoroTextNormalizer.numero_para_palavras(int(m.group(1).replace(',', '').split('.')[0]))} euros",
                      texto)
        
        return texto
    
    @staticmethod
    def processar_numeros_especiais(texto: str) -> str:
        """Processa códigos, modelos e identificadores especiais."""
        
        # Agent 007 -> agent zero zero seven
        def converter_codigo(match):
            codigo = match.group(0)
            digitos = re.findall(r'\d', codigo)
            digitos_lidos = ' '.join([KokoroTextNormalizer.UNIDADES[d] for d in digitos])
            prefixo = re.sub(r'\d', '', codigo).strip()
            return f"{prefixo} {digitos_lidos}".strip()
        
        texto = re.sub(r'\b[A-Za-z]*\d+[A-Za-z]?\b', converter_codigo, texto)
        
        return texto
    
    @staticmethod
    def processar_romanos(texto: str) -> str:
        """Processa numerais romanos."""
        
        for romano, palavras in KokoroTextNormalizer.ROMANOS.items():
            # Louis XIV -> Louis the Fourteenth
            padrao = rf'\b{romano}\b'
            texto = re.sub(padrao, palavras, texto)
        
        return texto
    
    @staticmethod
    def processar_razoes_proporcoes(texto: str) -> str:
        """Processa razões e proporções."""
        
        # 16:9 -> sixteen by nine
        texto = re.sub(r'(\d+):(\d+)',
                      lambda m: f"{KokoroTextNormalizer.numero_para_palavras(int(m.group(1)))} by {KokoroTextNormalizer.numero_para_palavras(int(m.group(2)))}",
                      texto)
        
        # 3 to 1 (já está em formato texto)
        
        return texto
    
    @staticmethod
    def normalizar(texto: str) -> str:
        """
        Normaliza texto preservando toda informação.
        Ordem é importante: fazer operações mais específicas antes das mais gerais.
        """
        
        # Preservar estrutura original com marcadores
        original_length = len(texto)
        
        # 1. Operações matemáticas (mais específicas)
        texto = KokoroTextNormalizer.processar_operacoes_matematicas(texto)
        
        # 2. Valores monetários
        texto = KokoroTextNormalizer.processar_valores_monetarios(texto)
        
        # 3. Unidades de medida
        texto = KokoroTextNormalizer.processar_unidades_medida(texto)
        
        # 4. Datas
        texto = KokoroTextNormalizer.processar_datas(texto)
        
        # 5. Percentuais
        texto = KokoroTextNormalizer.processar_percentuais(texto)
        
        # 6. Frações
        texto = KokoroTextNormalizer.processar_fracoes(texto)
        
        # 7. Decimais
        texto = KokoroTextNormalizer.processar_decimais(texto)
        
        # 8. Romanos
        texto = KokoroTextNormalizer.processar_romanos(texto)
        
        # 9. Razões
        texto = KokoroTextNormalizer.processar_razoes_proporcoes(texto)
        
        # 10. Números especiais (códigos, modelos)
        texto = KokoroTextNormalizer.processar_numeros_especiais(texto)
        
        # 11. Números gerais (por último)
        texto = KokoroTextNormalizer.processar_numeros(texto)
        
        # Sanidade: verificar se perdemos caracteres
        texto_sem_espacos = texto.replace(' ', '')
        if len(texto_sem_espacos) < original_length * 0.8:  # Se ficou muito mais curto, pode ter problema
            print(f"⚠️ Aviso: Texto pode ter perdido informação. Original: {original_length}, Normalizado: {len(texto_sem_espacos)}")
        
        return texto


# ============= EXEMPLOS DE USO =============

if __name__ == "__main__":
    normalizer = KokoroTextNormalizer()
    
    exemplos = [
        "He bought 2 swords and paid $50.99 for them.",
        "The journey lasted 15 days and 3 hours.",
        "There were 47 villagers in the square.",
        "The book contained 101 spells.",
        "The dragon was 386 years old.",
        "The reward was 1,500 gold coins.",
        "The story takes place in the year 2025.",
        "He was the 1st to arrive.",
        "The festival was in its 23rd year.",
        "There was only a 1% chance of failure.",
        "The shield blocked 15% of the damage.",
        "She used 50% of her mana.",
        "Two plus three equals five: 2 + 3 = 5",
        "He weighed 5 kilos and was 1.75 meters tall.",
        "The temperature was 15°C and distance 10 km.",
        "The holographic screen displayed 16:9 ratio.",
        "The home team was winning 3 to 1.",
        "Louis XIV ruled for many years.",
        "Agent 007 received protocol 87-B.",
        "He drank 1/2 of the potion.",
        "R$ 50 for the daily rate and € 100 for the artifact.",
    ]
    
    print("=" * 80)
    print("KOKORO TEXT NORMALIZER - EXEMPLOS")
    print("=" * 80)
    
    for exemplo in exemplos:
        normalizado = normalizer.normalizar(exemplo)
        print(f"\nORIGINAL:\n  {exemplo}")
        print(f"NORMALIZADO:\n  {normalizado}")
        print("-" * 80)
