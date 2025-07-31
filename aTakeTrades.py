import tkinter as tk
from tkinter import messagebox, ttk
import MetaTrader5 as mt5
import threading
from PIL import ImageGrab, ImageFilter, ImageTk

# ===================== MT5 Trading Class =====================
class MT5Trader:
    MAGIC_NUMBER = 123456

    def __init__(self):
        if not mt5.initialize():
            raise SystemExit(f"MT5 initialization failed: {mt5.last_error()}")

    def shutdown(self):
        mt5.shutdown()

    def get_tick_info(self, symbol):
        tick = mt5.symbol_info_tick(symbol)
        symbol_info = mt5.symbol_info(symbol)
        if not tick or not symbol_info:
            return None, None
        return tick, symbol_info

    def calculate_lot_size(self, symbol, sl_points, risk_percent):
        account_info = mt5.account_info()
        if account_info is None:
            return 0.0

        balance = account_info.balance
        risk_amount = (risk_percent / 100) * balance

        tick, symbol_info = self.get_tick_info(symbol)
        if not tick or not symbol_info:
            return 0.0

        point = symbol_info.point
        tick_value = symbol_info.trade_tick_value

        sl_price_range = sl_points * point
        sl_value_per_lot = sl_price_range * (tick_value / point)
        if sl_value_per_lot == 0:
            return 0.0

        lot = risk_amount / sl_value_per_lot
        min_lot = symbol_info.volume_min
        max_lot = symbol_info.volume_max
        lot_step = symbol_info.volume_step

        lot = max(min_lot, min(lot, max_lot))
        lot = round(lot / lot_step) * lot_step
        lot = round(lot, 2)

        return lot if lot >= min_lot else 0.0

    def place_order_with_tp(self, symbol, order_type, lot, sl_points, rr_ratio=3.0, comment="AutoTrade"):
        tick, symbol_info = self.get_tick_info(symbol)
        if not tick or not symbol_info:
            return False

        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
        point = symbol_info.point
        sl = sl_points * point

        sl_price = price - sl if order_type == mt5.ORDER_TYPE_BUY else price + sl
        tp_price = price + (rr_ratio * sl) if order_type == mt5.ORDER_TYPE_BUY else price - (rr_ratio * sl)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 5,
            "magic": self.MAGIC_NUMBER,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        return result.retcode == mt5.TRADE_RETCODE_DONE

    def resolve_symbol_case(self, user_input):
        for sym in mt5.symbols_get():
            if sym.name.lower() == user_input.lower():
                return sym.name
        return None

# ===================== GUI Class =====================
class TradeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🚀 Pro Trading Terminal v3.9")
        self.root.configure(bg="#0E0E0E")
        self.trade_direction = None

        # --- Glass effect background capture ---
        self.bg_image = self.create_blurred_bg()
        self.bg_label = tk.Label(root, image=self.bg_image)
        self.bg_label.place(relwidth=1, relheight=1)

        # --- Compact panel for better spacing ---
        self.panel = tk.Frame(root, bg="#FFFFFF", highlightbackground="#CCC", highlightthickness=1)
        self.panel.place(relx=0.5, rely=0.5, anchor="center", width=400, height=320)

        font_label = ("Segoe UI", 10, "bold")
        paddings = {'padx': 8, 'pady': 4}
        entry_style = {"bg": "#f9f9f9", "relief": "solid", "bd": 1}

        # Symbol
        tk.Label(self.panel, text="Symbol:", font=font_label, bg="white", fg="#222").grid(row=0, column=0, sticky='w', **paddings)
        self.symbol_entry = tk.Entry(self.panel, **entry_style, width=18)
        self.symbol_entry.grid(row=0, column=1, **paddings)

        # SL Mode
        tk.Label(self.panel, text="SL Type:", font=font_label, bg="white", fg="#222").grid(row=1, column=0, sticky='w', **paddings)
        self.sl_mode_var = tk.StringVar(value="Points")
        self.sl_mode_dropdown = ttk.Combobox(self.panel, textvariable=self.sl_mode_var, values=["Points", "Price"], state="readonly", width=15)
        self.sl_mode_dropdown.grid(row=1, column=1, **paddings)

        # SL Value
        tk.Label(self.panel, text="SL Value:", font=font_label, bg="white", fg="#222").grid(row=2, column=0, sticky='w', **paddings)
        self.sl_entry = tk.Entry(self.panel, **entry_style, width=18)
        self.sl_entry.grid(row=2, column=1, **paddings)

        # Risk %
        tk.Label(self.panel, text="Risk %:", font=font_label, bg="white", fg="#222").grid(row=3, column=0, sticky='w', **paddings)
        self.risk_entry = tk.Entry(self.panel, **entry_style, width=18)
        self.risk_entry.insert(0, "0.25")
        self.risk_entry.grid(row=3, column=1, **paddings)

        # BUY / SELL buttons
        tk.Label(self.panel, text="Direction:", font=font_label, bg="white", fg="#222").grid(row=4, column=0, sticky='w', **paddings)
        btn_frame = tk.Frame(self.panel, bg="white")
        btn_frame.grid(row=4, column=1, pady=5)

        self.buy_btn = tk.Button(btn_frame, text="BUY", width=8, height=1, bg="#4CAF50", fg="white", font=("Segoe UI", 10, "bold"), relief="flat", command=lambda: self.set_direction("BUY"))
        self.buy_btn.pack(side="left", padx=5)

        self.sell_btn = tk.Button(btn_frame, text="SELL", width=8, height=1, bg="#E53935", fg="white", font=("Segoe UI", 10, "bold"), relief="flat", command=lambda: self.set_direction("SELL"))
        self.sell_btn.pack(side="left", padx=5)

        # Live info label
        self.info_label = tk.Label(self.panel, text="Price: -- | Spread: -- | Lot: --", font=("Segoe UI", 9), fg="#444", bg="white")
        self.info_label.grid(row=5, column=0, columnspan=2, pady=10)

        # Execute Trade button
        self.exec_btn = tk.Button(self.panel, text="🚀 Execute Trade", width=20, height=2, bg="#1e3c72", fg="white", font=("Segoe UI", 11, "bold"), relief="flat", command=self.submit_trade)
        self.exec_btn.grid(row=6, column=0, columnspan=2, pady=15)

        threading.Thread(target=self.update_live_info, daemon=True).start()

    def set_direction(self, direction):
        self.trade_direction = direction
        # Reset button styles
        self.buy_btn.config(bd=0, relief="flat")
        self.sell_btn.config(bd=0, relief="flat")

        # Apply black border to selected button
        if direction == "BUY":
            self.buy_btn.config(bd=3, relief="solid", highlightbackground="black")
        else:
            self.sell_btn.config(bd=3, relief="solid", highlightbackground="black")

    def create_blurred_bg(self):
        x = self.root.winfo_rootx()
        y = self.root.winfo_rooty()
        w = self.root.winfo_screenwidth()
        h = self.root.winfo_screenheight()
        screenshot = ImageGrab.grab(bbox=(x, y, w, h))
        blurred = screenshot.filter(ImageFilter.GaussianBlur(15))
        return ImageTk.PhotoImage(blurred)

    def update_live_info(self):
        while True:
            try:
                user_symbol = self.symbol_entry.get().strip()
                sl_mode = self.sl_mode_var.get().strip()
                sl_val = self.sl_entry.get().strip()
                risk_val = self.risk_entry.get().strip()
                if user_symbol:
                    trader = MT5Trader()
                    resolved = trader.resolve_symbol_case(user_symbol)
                    if resolved:
                        tick, info = trader.get_tick_info(resolved)
                        if tick and info:
                            price = tick.ask
                            spread = (tick.ask - tick.bid) / info.point
                            sl_points = 50
                            if sl_val:
                                sl_points = float(sl_val) if sl_mode == "Points" else abs((tick.ask - float(sl_val)) / info.point)
                            lot = trader.calculate_lot_size(resolved, sl_points, float(risk_val or 0.25))
                            self.info_label.config(text=f"Price: {price:.5f} | Spread: {spread:.1f} pts | Lot: {lot}")
                    trader.shutdown()
            except:
                pass
            self.info_label.update()
            self.info_label.after(1000)

    def submit_trade(self):
        user_input_symbol = self.symbol_entry.get().strip()
        if not self.trade_direction:
            messagebox.showerror("Input Error", "Please select BUY or SELL direction.")
            return
        sl_mode = self.sl_mode_var.get().strip()
        try:
            sl_value = float(self.sl_entry.get().strip())
            risk_percent = float(self.risk_entry.get().strip())
        except ValueError:
            messagebox.showerror("Input Error", "Invalid SL or Risk % value.")
            return
        trader = MT5Trader()
        resolved_symbol = trader.resolve_symbol_case(user_input_symbol)
        if not resolved_symbol:
            messagebox.showerror("Symbol Error", f"Symbol '{user_input_symbol}' not found.")
            trader.shutdown()
            return
        tick, symbol_info = trader.get_tick_info(resolved_symbol)
        if not tick:
            messagebox.showerror("MT5 Error", "Could not retrieve symbol info.")
            trader.shutdown()
            return
        if sl_mode == "Price":
            sl_points = abs((tick.ask - sl_value) / symbol_info.point) if self.trade_direction == "BUY" else abs((sl_value - tick.bid) / symbol_info.point)
        else:
            sl_points = sl_value
        order_type = mt5.ORDER_TYPE_BUY if self.trade_direction == "BUY" else mt5.ORDER_TYPE_SELL
        lot = trader.calculate_lot_size(resolved_symbol, sl_points, risk_percent)
        if lot <= 0:
            messagebox.showwarning("Lot Error", "Lot size is zero or invalid.")
            trader.shutdown()
            return
        success = trader.place_order_with_tp(resolved_symbol, order_type, lot, sl_points)
        trader.shutdown()
        if success:
            messagebox.showinfo("Success ✅", f"{resolved_symbol} {self.trade_direction} placed with SL & 1:3 TP.")
        else:
            messagebox.showerror("Trade Failed ❌", "Trade execution failed.")

# ===================== Center Window =====================
def center_window(window, width=460, height=360):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    window.geometry(f'{width}x{height}+{x}+{y}')

# ===================== Run App =====================
if __name__ == "__main__":
    app = tk.Tk()
    center_window(app, 460, 360)
    TradeGUI(app)
    app.mainloop()
