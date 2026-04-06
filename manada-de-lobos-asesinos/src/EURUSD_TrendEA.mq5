//+------------------------------------------------------------------+
//|                                                EURUSD_TrendEA.mq5|
//|                        Ejemplo educativo - no garantiza beneficios|
//+------------------------------------------------------------------+
#property strict
#property description "EA para EURUSD con lectura M5 + confirmación M15"

#include <Trade/Trade.mqh>

input string   InpSymbol                 = "EURUSD";
input ENUM_TIMEFRAMES InpEntryTF         = PERIOD_M5;   // Lectura principal (5 min)
input ENUM_TIMEFRAMES InpTrendTF         = PERIOD_M15;  // Confirmación de tendencia (15 min)
input double   InpRiskPercent            = 1.0;         // Riesgo por operación (%)
input int      InpFastEMA                = 50;
input int      InpSlowEMA                = 200;
input int      InpATRPeriod              = 14;
input double   InpSL_ATR_Mult            = 1.8;
input double   InpTP_ATR_Mult            = 3.0;
input int      InpBreakoutLookback       = 20;
input int      InpMaxSpreadPoints        = 25;
input int      InpMagic                  = 260406;
input bool     InpOnePositionOnly        = true;

CTrade trade;
int hFast = INVALID_HANDLE;
int hSlow = INVALID_HANDLE;
int hATR  = INVALID_HANDLE;
datetime lastBarTime = 0;

bool IsAllowedTF(const ENUM_TIMEFRAMES tf)
{
   return (tf == PERIOD_M5 || tf == PERIOD_M15);
}

bool IsNewBar(const string symbol, const ENUM_TIMEFRAMES tf)
{
   datetime t[];
   if(CopyTime(symbol, tf, 0, 1, t) < 1)
      return false;

   if(t[0] != lastBarTime)
   {
      lastBarTime = t[0];
      return true;
   }
   return false;
}

double NormalizeLots(const string symbol, double lots)
{
   double minLot  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double maxLot  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   double stepLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);

   lots = MathMax(minLot, MathMin(maxLot, lots));
   lots = MathFloor(lots / stepLot) * stepLot;
   return NormalizeDouble(lots, 2);
}

double CalcLotsByRisk(const string symbol, double stopDistancePrice)
{
   if(stopDistancePrice <= 0.0)
      return 0.0;

   double balance     = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskMoney   = balance * (InpRiskPercent / 100.0);
   double tickValue   = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize    = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);

   if(tickValue <= 0.0 || tickSize <= 0.0)
      return 0.0;

   double valuePerPriceUnit = tickValue / tickSize;
   double lots = riskMoney / (stopDistancePrice * valuePerPriceUnit);

   return NormalizeLots(symbol, lots);
}

bool HasOpenPosition(const string symbol)
{
   for(int i = PositionsTotal() - 1; i >= 0; --i)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;

      if(PositionSelectByTicket(ticket))
      {
         string s = PositionGetString(POSITION_SYMBOL);
         long mg  = PositionGetInteger(POSITION_MAGIC);
         if(s == symbol && mg == InpMagic)
            return true;
      }
   }
   return false;
}

int OnInit()
{
   if(_Symbol != InpSymbol)
      Print("Advertencia: el EA fue diseñado para EURUSD. Símbolo actual: ", _Symbol);

   if(!IsAllowedTF(InpEntryTF) || !IsAllowedTF(InpTrendTF))
   {
      Print("Error: este EA solo permite timeframes M5 o M15.");
      return INIT_FAILED;
   }

   hFast = iMA(InpSymbol, InpTrendTF, InpFastEMA, 0, MODE_EMA, PRICE_CLOSE);
   hSlow = iMA(InpSymbol, InpTrendTF, InpSlowEMA, 0, MODE_EMA, PRICE_CLOSE);
   hATR  = iATR(InpSymbol, InpEntryTF, InpATRPeriod);

   if(hFast == INVALID_HANDLE || hSlow == INVALID_HANDLE || hATR == INVALID_HANDLE)
      return INIT_FAILED;

   trade.SetExpertMagicNumber(InpMagic);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   if(hFast != INVALID_HANDLE) IndicatorRelease(hFast);
   if(hSlow != INVALID_HANDLE) IndicatorRelease(hSlow);
   if(hATR  != INVALID_HANDLE) IndicatorRelease(hATR);
}

void OnTick()
{
   if(_Symbol != InpSymbol)
      return;

   if(!IsNewBar(InpSymbol, InpEntryTF))
      return;

   double spreadPts = (double)SymbolInfoInteger(InpSymbol, SYMBOL_SPREAD);
   if(spreadPts > InpMaxSpreadPoints)
      return;

   if(InpOnePositionOnly && HasOpenPosition(InpSymbol))
      return;

   double fast[3], slow[3], atr[3];
   if(CopyBuffer(hFast, 0, 0, 3, fast) < 3) return;
   if(CopyBuffer(hSlow, 0, 0, 3, slow) < 3) return;
   if(CopyBuffer(hATR,  0, 0, 3, atr)  < 3) return;

   MqlRates rates[];
   if(CopyRates(InpSymbol, InpEntryTF, 0, InpBreakoutLookback + 3, rates) < InpBreakoutLookback + 3)
      return;

   ArraySetAsSeries(rates, true);

   bool trendUp = fast[1] > slow[1];
   bool trendDn = fast[1] < slow[1];

   double highest = rates[2].high;
   double lowest  = rates[2].low;
   for(int i = 2; i <= InpBreakoutLookback + 1; ++i)
   {
      highest = MathMax(highest, rates[i].high);
      lowest  = MathMin(lowest, rates[i].low);
   }

   double ask = SymbolInfoDouble(InpSymbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(InpSymbol, SYMBOL_BID);
   double atrValue = atr[1];

   if(atrValue <= 0.0)
      return;

   // Compra: tendencia M15 alcista + ruptura en M5/M15 (según InpEntryTF)
   if(trendUp && rates[1].close > highest)
   {
      double sl = ask - (InpSL_ATR_Mult * atrValue);
      double tp = ask + (InpTP_ATR_Mult * atrValue);
      double lots = CalcLotsByRisk(InpSymbol, ask - sl);

      if(lots > 0.0)
         trade.Buy(lots, InpSymbol, ask, sl, tp, "EURUSD M5/M15 trend breakout BUY");
      return;
   }

   // Venta: tendencia M15 bajista + ruptura en M5/M15 (según InpEntryTF)
   if(trendDn && rates[1].close < lowest)
   {
      double sl = bid + (InpSL_ATR_Mult * atrValue);
      double tp = bid - (InpTP_ATR_Mult * atrValue);
      double lots = CalcLotsByRisk(InpSymbol, sl - bid);

      if(lots > 0.0)
         trade.Sell(lots, InpSymbol, bid, sl, tp, "EURUSD M5/M15 trend breakout SELL");
   }
}
//+------------------------------------------------------------------+
